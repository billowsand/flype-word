#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui;
use egui_extras::{Column, TableBuilder};
use flype_word::{
    EncodedEntry, FrequencyEntry, WordEntry, encode_entries, extract_frequencies, find_conflicts,
    parse_word_list_lenient, render_conflicts, render_encoded,
};
use jieba_rs::Jieba;
use rfd::FileDialog;
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, TryRecvError};

const APP_TITLE: &str = "小鹤双拼词库生成器";
const SFSS_FONT: &[u8] = include_bytes!("../../font/sfss.ttf");
/// 预览框最多显示多少行；导出的码表始终是完整的。
const PREVIEW_MAX_LINES: usize = 2000;
/// “操作”列的最小宽度：必须装得下 ↑ ↓ 删除 三个按钮，由测试兜底。
const ACTION_COLUMN_MIN_WIDTH: f32 = 166.0;

const OK_COLOR: egui::Color32 = egui::Color32::from_rgb(24, 140, 76);
const ERROR_COLOR: egui::Color32 = egui::Color32::from_rgb(200, 70, 54);
const WARNING_COLOR: egui::Color32 = egui::Color32::from_rgb(176, 112, 16);

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title(APP_TITLE)
            .with_inner_size([1360.0, 860.0])
            .with_min_inner_size([960.0, 640.0]),
        centered: true,
        ..Default::default()
    };

    eframe::run_native(
        APP_TITLE,
        options,
        Box::new(|creation_context| Ok(Box::new(DictionaryApp::new(&creation_context.egui_ctx)))),
    )
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Screen {
    #[default]
    Source,
    Words,
    Output,
}

/// 一次分词用到的全部参数，用来判断界面上的设置是否已经和当前词表脱节。
#[derive(Clone, Debug, PartialEq, Eq)]
struct SegmentOptions {
    min_frequency: u64,
    include_single_chars: bool,
    hmm: bool,
    user_dict_path: Option<PathBuf>,
}

/// 后台分词线程的结果。`Jieba` 会一并交回主线程缓存，避免每次点击都重新建词典。
enum SegmentOutcome {
    Ready {
        jieba: Arc<Jieba>,
        entries: Vec<FrequencyEntry>,
        options: SegmentOptions,
    },
    Failed(String),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct EditableWord {
    /// 与行位置无关的稳定标识，用来固定单元格的控件状态（光标、选区、输入法）。
    id: u64,
    word: String,
    frequency: String,
    pinyin: String,
    code: String,
    error: Option<String>,
    /// 上次编码时的输入快照，命中即复用 `cached`，这样按键只重算被改的那一行。
    cache_key: String,
    cached: Option<Result<EncodedEntry, String>>,
}

impl EditableWord {
    fn new(id: u64) -> Self {
        Self {
            id,
            ..Default::default()
        }
    }

    fn from_word_entry(id: u64, entry: WordEntry) -> Self {
        Self {
            id,
            word: entry.word,
            frequency: entry
                .frequency
                .map(|value| value.to_string())
                .unwrap_or_default(),
            pinyin: entry
                .pinyin
                .map(|value| value.join(" "))
                .unwrap_or_default(),
            ..Default::default()
        }
    }

    fn cache_key(&self) -> String {
        format!("{}\u{1}{}\u{1}{}", self.word, self.frequency, self.pinyin)
    }

    /// 只在词语、词频或拼音变化后重新编码。
    fn refresh_encoding(&mut self) {
        let key = self.cache_key();
        if self.cached.is_some() && self.cache_key == key {
            return;
        }
        let result = editable_to_entry(self)
            .and_then(|entry| encode_entries(&[entry]).map_err(|error| error.to_string()))
            .map(|mut encoded| encoded.remove(0));
        self.cached = Some(result);
        self.cache_key = key;
    }
}

struct DictionaryApp {
    screen: Screen,
    source_text: String,
    source_path: Option<PathBuf>,
    user_dict_path: Option<PathBuf>,
    min_frequency: u64,
    include_single_chars: bool,
    hmm: bool,
    words: Vec<EditableWord>,
    next_word_id: u64,
    encoded: Vec<EncodedEntry>,
    conflicts: BTreeMap<String, Vec<String>>,
    output_text: String,
    with_flypy_header: bool,
    search: String,
    only_errors: bool,
    status: String,
    status_is_error: bool,
    dark_mode: bool,
    /// 缓存的 jieba 及它已经加载的用户词典；换词典才需要重建。
    jieba: Option<Arc<Jieba>>,
    jieba_dict_path: Option<PathBuf>,
    /// 正在运行的后台分词任务。
    segmentation: Option<Receiver<SegmentOutcome>>,
    /// 当前词表是用哪套参数分出来的。
    words_options: Option<SegmentOptions>,
    /// 词表是否有未保存的人工修订。
    unsaved: bool,
    exit_prompt: bool,
    exit_confirmed: bool,
    /// 中央面板上一帧的内容范围与可用范围，供布局回归测试断言内容没有溢出面板。
    #[cfg(test)]
    last_content_rect: egui::Rect,
    #[cfg(test)]
    last_panel_rect: egui::Rect,
}

impl DictionaryApp {
    fn new(context: &egui::Context) -> Self {
        install_font(context);
        // 用 set_theme 而不是 set_visuals：后者会写进“当前跟随系统的那个主题”的样式，
        // 系统主题一变就和界面上的开关状态对不上。
        context.set_theme(egui::ThemePreference::Light);

        configure_style(context, egui::Theme::Light);
        configure_style(context, egui::Theme::Dark);

        Self {
            screen: Screen::Source,
            source_text: String::new(),
            source_path: None,
            user_dict_path: None,
            min_frequency: 1,
            include_single_chars: false,
            hmm: true,
            words: Vec::new(),
            next_word_id: 0,
            encoded: Vec::new(),
            conflicts: BTreeMap::new(),
            output_text: String::new(),
            with_flypy_header: true,
            search: String::new(),
            only_errors: false,
            status: "请打开文本文件、拖入文件，或直接粘贴原文。".to_owned(),
            status_is_error: false,
            dark_mode: false,
            jieba: None,
            jieba_dict_path: None,
            segmentation: None,
            words_options: None,
            unsaved: false,
            exit_prompt: false,
            exit_confirmed: false,
            #[cfg(test)]
            last_content_rect: egui::Rect::NOTHING,
            #[cfg(test)]
            last_panel_rect: egui::Rect::NOTHING,
        }
    }

    fn set_success(&mut self, message: impl Into<String>) {
        self.status = message.into();
        self.status_is_error = false;
    }

    fn set_error(&mut self, message: impl Into<String>) {
        self.status = message.into();
        self.status_is_error = true;
    }

    fn take_word_id(&mut self) -> u64 {
        let id = self.next_word_id;
        self.next_word_id += 1;
        id
    }

    fn current_options(&self) -> SegmentOptions {
        SegmentOptions {
            min_frequency: self.min_frequency,
            include_single_chars: self.include_single_chars,
            hmm: self.hmm,
            user_dict_path: self.user_dict_path.clone(),
        }
    }

    /// 界面上的参数是否已经和当前词表对应的参数不一致。
    fn options_are_stale(&self) -> bool {
        self.words_options
            .as_ref()
            .is_some_and(|options| *options != self.current_options())
    }

    fn open_source_file(&mut self) {
        let Some(path) = FileDialog::new()
            .set_title("打开中文文本")
            .add_filter("文本文件", &["txt", "md", "text"])
            .add_filter("所有文件", &["*"])
            .pick_file()
        else {
            return;
        };
        self.load_source(&path);
    }

    fn load_source(&mut self, path: &Path) {
        match read_text_file(path) {
            Ok((text, encoding)) => {
                self.source_text = text;
                self.source_path = Some(path.to_path_buf());
                self.screen = Screen::Source;
                if encoding == "UTF-8" {
                    self.set_success(format!("已打开：{}", path.display()));
                } else {
                    self.set_success(format!("已按 {encoding} 解码打开：{}", path.display()));
                }
            }
            Err(error) => self.set_error(error),
        }
    }

    fn choose_user_dict(&mut self) {
        if let Some(path) = FileDialog::new()
            .set_title("选择 jieba 用户词典")
            .add_filter("词典或文本文件", &["txt", "dict"])
            .add_filter("所有文件", &["*"])
            .pick_file()
        {
            self.user_dict_path = Some(path.clone());
            self.set_success(format!("已选择用户词典：{}", path.display()));
        }
    }

    fn clear_source(&mut self) {
        self.source_text.clear();
        self.source_path = None;
        self.words.clear();
        self.encoded.clear();
        self.conflicts.clear();
        self.output_text.clear();
        self.words_options = None;
        self.search.clear();
        self.only_errors = false;
        self.unsaved = false;
        self.screen = Screen::Source;
        self.set_success("已清空原文和已生成的词表。");
    }

    /// 分词放到后台线程：`Jieba::new()` 和长文本切分都是几百毫秒级别的活，
    /// 留在 UI 线程上窗口会假死。
    fn start_segmentation(&mut self, ctx: &egui::Context) {
        if self.segmentation.is_some() {
            return;
        }
        if self.source_text.trim().is_empty() {
            self.set_error("原文为空，无法分词。");
            return;
        }

        let options = self.current_options();
        let cached = self
            .jieba
            .clone()
            .filter(|_| self.jieba_dict_path == options.user_dict_path);
        let text = self.source_text.clone();
        let (sender, receiver) = mpsc::channel();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let _ = sender.send(segment_in_background(cached, &options, &text));
            ctx.request_repaint();
        });

        self.segmentation = Some(receiver);
        self.set_success("正在分词，请稍候……");
    }

    fn poll_segmentation(&mut self) {
        let Some(receiver) = self.segmentation.take() else {
            return;
        };
        match receiver.try_recv() {
            Err(TryRecvError::Empty) => self.segmentation = Some(receiver),
            Err(TryRecvError::Disconnected) => self.set_error("分词任务意外结束，请重试。"),
            Ok(SegmentOutcome::Failed(error)) => self.set_error(error),
            Ok(SegmentOutcome::Ready {
                jieba,
                entries,
                options,
            }) => {
                self.jieba = Some(jieba);
                self.jieba_dict_path = options.user_dict_path.clone();
                self.words_options = Some(options);
                self.replace_words(
                    entries
                        .into_iter()
                        .map(|entry| WordEntry {
                            word: entry.word,
                            frequency: Some(entry.frequency),
                            pinyin: None,
                        })
                        .collect(),
                );
                self.screen = Screen::Words;
                self.set_success(format!("分词完成，共得到 {} 个词语。", self.words.len()));
            }
        }
    }

    fn replace_words(&mut self, entries: Vec<WordEntry>) {
        let mut next_id = self.next_word_id;
        let rows: Vec<EditableWord> = entries
            .into_iter()
            .map(|entry| {
                let id = next_id;
                next_id += 1;
                EditableWord::from_word_entry(id, entry)
            })
            .collect();
        self.next_word_id = next_id;
        self.words = rows;
        self.rebuild_output();
        self.unsaved = !self.words.is_empty();
    }

    fn import_word_list(&mut self) {
        let Some(path) = FileDialog::new()
            .set_title("打开人工修订词表")
            .add_filter("TSV 或文本文件", &["tsv", "txt"])
            .add_filter("所有文件", &["*"])
            .pick_file()
        else {
            return;
        };
        self.load_word_list(&path);
    }

    fn load_word_list(&mut self, path: &Path) {
        match read_text_file(path) {
            // 用宽松解析：重复词语显示成行错误，当场就能改，不必整份文件退回。
            Ok((text, _)) => {
                self.replace_words(parse_word_list_lenient(&text));
                self.words_options = None;
                self.screen = Screen::Words;
                let error_count = self.error_count();
                if error_count > 0 {
                    self.only_errors = true;
                    self.set_error(format!(
                        "已导入 {} 个词语，其中 {error_count} 行有问题（已切换到“只看错误”）：{}",
                        self.words.len(),
                        path.display()
                    ));
                } else {
                    self.set_success(format!(
                        "已导入 {} 个词语：{}",
                        self.words.len(),
                        path.display()
                    ));
                }
            }
            Err(error) => self.set_error(error),
        }
    }

    fn save_editable_word_list(&mut self) {
        let Some(path) = FileDialog::new()
            .set_title("保存人工修订词表")
            .add_filter("TSV 文件", &["tsv"])
            .set_file_name("words.tsv")
            .save_file()
        else {
            return;
        };

        match write_utf8_bom_file(&path, &render_editable_words(&self.words)) {
            Ok(()) => {
                self.unsaved = false;
                self.set_success(format!("词表已保存：{}", path.display()));
            }
            Err(error) => self.set_error(error),
        }
    }

    fn export_dictionary(&mut self) {
        self.rebuild_output();
        let error_count = self.error_count();
        if error_count > 0 {
            self.set_error(format!(
                "仍有 {error_count} 行错误，请修正后再导出最终码表。"
            ));
            self.screen = Screen::Words;
            self.only_errors = true;
            return;
        }
        if self.encoded.is_empty() {
            self.set_error("当前没有可导出的词语。");
            return;
        }

        let Some(path) = FileDialog::new()
            .set_title("导出小鹤双拼用户码表")
            .add_filter("文本文件", &["txt"])
            .set_file_name("flypy-user.txt")
            .save_file()
        else {
            return;
        };

        match write_utf8_bom_file(&path, &self.output_text) {
            Ok(()) => {
                self.unsaved = false;
                self.set_success(format!(
                    "已导出 {} 个词语：{}",
                    self.encoded.len(),
                    path.display()
                ));
            }
            Err(error) => self.set_error(error),
        }
    }

    fn export_conflicts(&mut self) {
        let Some(path) = FileDialog::new()
            .set_title("保存重码报告")
            .add_filter("TSV 文件", &["tsv"])
            .set_file_name("conflicts.tsv")
            .save_file()
        else {
            return;
        };

        match write_utf8_bom_file(&path, &render_conflicts(&self.conflicts)) {
            Ok(()) => self.set_success(format!("重码报告已保存：{}", path.display())),
            Err(error) => self.set_error(error),
        }
    }

    /// 把第 `from` 行挪到 `to` 的位置。`to` 传的是相邻**可见**行的下标，
    /// 这样筛选状态下上下移动也是按屏幕上看到的顺序走。
    fn move_row(&mut self, from: usize, to: usize) {
        if from >= self.words.len() {
            return;
        }
        let row = self.words.remove(from);
        self.words.insert(to.min(self.words.len()), row);
    }

    fn error_count(&self) -> usize {
        self.words
            .iter()
            .filter(|word| word.error.is_some())
            .count()
    }

    fn rebuild_output(&mut self) {
        let mut word_counts = HashMap::<&str, usize>::new();
        for row in &self.words {
            let word = row.word.trim();
            if !word.is_empty() {
                *word_counts.entry(word).or_default() += 1;
            }
        }
        let duplicated: Vec<bool> = self
            .words
            .iter()
            .map(|row| {
                word_counts
                    .get(row.word.trim())
                    .copied()
                    .unwrap_or_default()
                    > 1
            })
            .collect();

        self.encoded.clear();
        for (row, is_duplicate) in self.words.iter_mut().zip(duplicated) {
            row.code.clear();
            row.error = None;

            if row.word.trim().is_empty() {
                row.error = Some("词语不能为空".to_owned());
                continue;
            }
            if is_duplicate {
                row.error = Some("词语重复".to_owned());
                continue;
            }

            row.refresh_encoding();
            match row.cached.as_ref().expect("刚刷新过编码缓存") {
                Ok(entry) => {
                    row.code.clone_from(&entry.code);
                    self.encoded.push(entry.clone());
                }
                Err(error) => row.error = Some(error.clone()),
            }
        }

        self.conflicts = find_conflicts(&self.encoded);
        self.output_text = render_encoded(&self.encoded, self.with_flypy_header);
    }

    /// 把拖进窗口的文件按扩展名分派：`.tsv` 当词表，其它当原文。
    fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped: Option<PathBuf> = ctx.input(|input| {
            input
                .raw
                .dropped_files
                .iter()
                .find_map(|file| file.path.clone())
        });
        let Some(path) = dropped else {
            return;
        };

        let is_word_list = path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("tsv"));
        if is_word_list || self.screen == Screen::Words {
            self.load_word_list(&path);
        } else {
            self.load_source(&path);
        }
    }

    fn handle_close_request(&mut self, ctx: &egui::Context) {
        if ctx.input(|input| input.viewport().close_requested())
            && self.unsaved
            && !self.exit_confirmed
        {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.exit_prompt = true;
        }
    }

    fn render_exit_prompt(&mut self, ctx: &egui::Context) {
        if !self.exit_prompt {
            return;
        }

        let response = egui::Modal::new(egui::Id::new("exit-prompt")).show(ctx, |ui| {
            ui.set_width(380.0);
            ui.heading("词表尚未保存");
            ui.label("直接关闭会丢失当前的人工修订结果。");
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                if ui.button("保存后退出").clicked() {
                    self.save_editable_word_list();
                    if !self.unsaved {
                        self.exit_prompt = false;
                        self.exit_confirmed = true;
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                }
                if ui.button("直接退出").clicked() {
                    self.exit_prompt = false;
                    self.exit_confirmed = true;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
                if ui.button("返回继续编辑").clicked() {
                    self.exit_prompt = false;
                }
            });
        });

        if response.should_close() {
            self.exit_prompt = false;
        }
    }

    fn render_navigation(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading(APP_TITLE);
            ui.separator();
            ui.selectable_value(&mut self.screen, Screen::Source, "1  原文与分词");
            ui.selectable_value(&mut self.screen, Screen::Words, "2  人工修订");
            ui.selectable_value(&mut self.screen, Screen::Output, "3  最终码表");

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let theme_label = if self.dark_mode {
                    "切换浅色"
                } else {
                    "切换深色"
                };
                if ui.button(theme_label).clicked() {
                    self.dark_mode = !self.dark_mode;
                    ui.ctx().set_theme(if self.dark_mode {
                        egui::ThemePreference::Dark
                    } else {
                        egui::ThemePreference::Light
                    });
                }
                if self.unsaved {
                    ui.label("● 词表未保存");
                }
            });
        });
    }

    fn render_source(&mut self, ui: &mut egui::Ui) {
        let running = self.segmentation.is_some();

        ui.horizontal(|ui| {
            if ui.button("打开文本…").clicked() {
                self.open_source_file();
            }
            if ui
                .button("清空")
                .on_hover_text("清空原文，并丢弃已生成的词表和码表")
                .clicked()
            {
                self.clear_source();
            }
            ui.separator();
            ui.label("最低词频");
            ui.add(
                egui::DragValue::new(&mut self.min_frequency)
                    .range(1..=9999)
                    .speed(0.1),
            );
            ui.checkbox(&mut self.include_single_chars, "保留单字");
            ui.checkbox(&mut self.hmm, "HMM 新词识别");
        });

        ui.horizontal(|ui| {
            if ui.button("选择 jieba 用户词典…").clicked() {
                self.choose_user_dict();
            }
            let dictionary = self
                .user_dict_path
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "未选择（使用内置词典）".to_owned());
            ui.label(dictionary);
            if self.user_dict_path.is_some() && ui.small_button("移除").clicked() {
                self.user_dict_path = None;
            }
        });

        ui.separator();
        ui.horizontal(|ui| {
            let path_label = self
                .source_path
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "可在下方直接粘贴中文原文，或把 txt 文件拖进窗口".to_owned());
            ui.label(path_label);
            if self.options_are_stale() {
                ui.colored_label(WARNING_COLOR, "参数已修改，需重新分词后生效");
            }
        });

        // 先把底部按钮行的高度留出来，编辑区只用剩下的空间并且自己滚动，
        // 否则文本一多，TextEdit 会一直长高，把按钮顶到窗口外面。
        let spacing = ui.spacing().item_spacing.y;
        let footer_height = 38.0 + spacing * 2.0;
        let editor_height = (ui.available_height() - footer_height).max(120.0);
        egui::ScrollArea::vertical()
            .id_salt("source-editor")
            .max_height(editor_height)
            .auto_shrink([false; 2])
            .show(ui, |ui| {
                ui.add_sized(
                    [ui.available_width(), editor_height],
                    egui::TextEdit::multiline(&mut self.source_text)
                        .hint_text("在这里粘贴需要生成词库的中文文本……")
                        .desired_width(f32::INFINITY),
                );
            });

        ui.horizontal(|ui| {
            ui.label(format!("{} 个字符", self.source_text.chars().count()));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add_enabled(
                        !running,
                        egui::Button::new("开始分词  →").min_size(egui::vec2(150.0, 38.0)),
                    )
                    .clicked()
                {
                    let ctx = ui.ctx().clone();
                    self.start_segmentation(&ctx);
                }
                if running {
                    ui.add(egui::Spinner::new());
                    ui.label("正在分词……");
                }
            });
        });
    }

    fn render_words(&mut self, ui: &mut egui::Ui) {
        let error_count = self.error_count();
        ui.horizontal(|ui| {
            if ui.button("导入词表…").clicked() {
                self.import_word_list();
            }
            if ui.button("保存修订词表…").clicked() {
                self.save_editable_word_list();
            }
            if ui.button("添加一行").clicked() {
                let id = self.take_word_id();
                self.words.push(EditableWord::new(id));
                self.rebuild_output();
                self.unsaved = true;
            }
            ui.separator();
            ui.label(format!("共 {} 行", self.words.len()));
            ui.colored_label(
                if error_count == 0 {
                    OK_COLOR
                } else {
                    ERROR_COLOR
                },
                format!("{error_count} 行错误"),
            );
            ui.label(format!("{} 组重码", self.conflicts.len()));

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("生成最终码表  →").clicked() {
                    self.rebuild_output();
                    self.screen = Screen::Output;
                }
            });
        });

        ui.horizontal(|ui| {
            ui.label("筛选");
            ui.add(
                egui::TextEdit::singleline(&mut self.search)
                    .hint_text("输入词语、拼音或编码")
                    .desired_width(260.0),
            );
            ui.checkbox(&mut self.only_errors, "只看错误");
            if ui.small_button("清除筛选").clicked() {
                self.search.clear();
                self.only_errors = false;
            }
            ui.label("多音字可在“人工拼音”列填写，如：chong qing");
        });

        if self.words.is_empty() {
            ui.add_space(20.0);
            ui.vertical_centered(|ui| {
                ui.label("还没有词语。可以回到第 1 页分词，或在这里导入 / 拖入一份 TSV 词表。");
            });
            return;
        }

        let search = self.search.trim().to_lowercase();
        let visible_indices: Vec<usize> = self
            .words
            .iter()
            .enumerate()
            .filter(|(_, row)| {
                (!self.only_errors || row.error.is_some())
                    && (search.is_empty()
                        || row.word.to_lowercase().contains(&search)
                        || row.pinyin.to_lowercase().contains(&search)
                        || row.code.contains(&search))
            })
            .map(|(index, _)| index)
            .collect();

        if visible_indices.is_empty() {
            ui.add_space(20.0);
            ui.vertical_centered(|ui| {
                ui.label("没有符合筛选条件的行。");
            });
            return;
        }

        let mut changed = false;
        let mut remove_index = None;
        let mut move_action = None;

        TableBuilder::new(ui)
            .striped(true)
            .resizable(true)
            .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
            .column(Column::initial(48.0).at_least(40.0))
            .column(Column::initial(170.0).at_least(100.0))
            .column(Column::initial(80.0).at_least(65.0))
            .column(Column::initial(220.0).at_least(130.0))
            .column(Column::initial(90.0).at_least(70.0))
            .column(Column::remainder().at_least(130.0))
            // ↑ ↓ 删除 三个按钮实测需要 156 px，列宽必须给够，否则“删除”会被窗口边缘切掉。
            .column(
                Column::initial(ACTION_COLUMN_MIN_WIDTH + 10.0).at_least(ACTION_COLUMN_MIN_WIDTH),
            )
            .header(32.0, |mut header| {
                for title in [
                    "序号",
                    "词语",
                    "词频",
                    "人工拼音（可选）",
                    "编码",
                    "状态",
                    "操作",
                ] {
                    header.col(|ui| {
                        ui.strong(title);
                    });
                }
            })
            .body(|body| {
                body.rows(34.0, visible_indices.len(), |mut table_row| {
                    let position = table_row.index();
                    let index = visible_indices[position];
                    // 上/下移动按可见顺序走：筛选状态下和隐藏行交换会让人以为按钮没反应。
                    let previous_visible = position
                        .checked_sub(1)
                        .map(|previous| visible_indices[previous]);
                    let next_visible = visible_indices.get(position + 1).copied();
                    let word = &mut self.words[index];
                    let row_id = word.id;

                    table_row.col(|ui| {
                        ui.label((index + 1).to_string());
                    });
                    // push_id 用行的稳定 id：egui_extras 的单元格 id 取的是“可见行号”，
                    // 不加这一层，删除或筛选后焦点会留在原位继续编辑另一个词。
                    table_row.col(|ui| {
                        changed |= ui
                            .push_id(row_id, |ui| {
                                ui.text_edit_singleline(&mut word.word).changed()
                            })
                            .inner;
                    });
                    table_row.col(|ui| {
                        changed |= ui
                            .push_id(row_id, |ui| {
                                ui.text_edit_singleline(&mut word.frequency).changed()
                            })
                            .inner;
                    });
                    table_row.col(|ui| {
                        changed |= ui
                            .push_id(row_id, |ui| {
                                ui.text_edit_singleline(&mut word.pinyin).changed()
                            })
                            .inner;
                    });
                    table_row.col(|ui| {
                        ui.monospace(&word.code);
                    });
                    table_row.col(|ui| {
                        if let Some(error) = &word.error {
                            ui.colored_label(ERROR_COLOR, error);
                        } else {
                            ui.colored_label(OK_COLOR, "有效");
                        }
                    });
                    table_row.col(|ui| {
                        ui.push_id(row_id, |ui| {
                            ui.horizontal(|ui| {
                                if ui
                                    .add_enabled(
                                        previous_visible.is_some(),
                                        egui::Button::new("↑").small(),
                                    )
                                    .clicked()
                                    && let Some(target) = previous_visible
                                {
                                    move_action = Some((index, target));
                                }
                                if ui
                                    .add_enabled(
                                        next_visible.is_some(),
                                        egui::Button::new("↓").small(),
                                    )
                                    .clicked()
                                    && let Some(target) = next_visible
                                {
                                    move_action = Some((index, target));
                                }
                                if ui.small_button("删除").clicked() {
                                    remove_index = Some(index);
                                }
                            });
                        });
                    });
                });
            });

        if let Some(index) = remove_index {
            self.words.remove(index);
            changed = true;
        }
        if let Some((from, to)) = move_action {
            self.move_row(from, to);
            changed = true;
        }
        if changed {
            self.rebuild_output();
            self.unsaved = true;
        }
    }

    fn render_output(&mut self, ui: &mut egui::Ui) {
        let error_count = self.error_count();
        ui.horizontal(|ui| {
            if ui
                .checkbox(&mut self.with_flypy_header, "包含小鹤用户码表配置头")
                .changed()
            {
                self.rebuild_output();
            }
            ui.separator();
            ui.label(format!("{} 个有效词语", self.encoded.len()));
            ui.label(format!("{} 组重码", self.conflicts.len()));
            if error_count > 0 {
                ui.colored_label(ERROR_COLOR, format!("{error_count} 行错误未包含在预览中"));
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("导出最终码表…").clicked() {
                    self.export_dictionary();
                }
                if ui.button("保存重码报告…").clicked() {
                    self.export_conflicts();
                }
                if ui.button("复制码表").clicked() {
                    ui.ctx().copy_text(self.output_text.clone());
                    self.status = format!("已复制 {} 个词语到剪贴板。", self.encoded.len());
                    self.status_is_error = false;
                }
            });
        });
        ui.separator();

        let (preview, truncated) = preview_slice(&self.output_text, PREVIEW_MAX_LINES);
        let conflicts = &self.conflicts;

        ui.columns(2, |columns| {
            columns[0].horizontal(|ui| {
                ui.heading("码表预览");
                if truncated {
                    ui.colored_label(
                        WARNING_COLOR,
                        format!("仅显示前 {PREVIEW_MAX_LINES} 行，导出和复制包含全部"),
                    );
                }
            });
            // 预览同样必须放进 ScrollArea：TextEdit 自己不滚动，几千行会长到十万像素高。
            egui::ScrollArea::vertical()
                .id_salt("dictionary-preview")
                .auto_shrink([false; 2])
                .show(&mut columns[0], |ui| {
                    // 传 &str 而不是 &mut String：只读但仍可选中复制。
                    let mut preview = preview;
                    ui.add(
                        egui::TextEdit::multiline(&mut preview)
                            .font(egui::TextStyle::Monospace)
                            .desired_width(f32::INFINITY),
                    );
                });

            columns[1].heading("重码报告");
            if conflicts.is_empty() {
                columns[1].colored_label(OK_COLOR, "当前没有重码。");
            } else {
                egui::ScrollArea::vertical()
                    .id_salt("conflicts")
                    .auto_shrink([false; 2])
                    .show(&mut columns[1], |ui| {
                        egui::Grid::new("conflicts-grid")
                            .striped(true)
                            .num_columns(2)
                            .show(ui, |ui| {
                                ui.strong("编码");
                                ui.strong("词语");
                                ui.end_row();
                                for (code, words) in conflicts {
                                    ui.monospace(code);
                                    ui.label(words.join("、"));
                                    ui.end_row();
                                }
                            });
                    });
            }
        });
    }

    fn render_status(&self, ui: &mut egui::Ui) {
        ui.separator();
        let color = if self.status_is_error {
            ERROR_COLOR
        } else {
            ui.visuals().weak_text_color()
        };
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(color, &self.status);
        });
    }
}

impl DictionaryApp {
    fn draw(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.poll_segmentation();
        self.handle_dropped_files(&ctx);
        self.handle_close_request(&ctx);

        egui::Panel::top("navigation").show(ui, |ui| {
            self.render_navigation(ui);
        });
        egui::Panel::bottom("status").show(ui, |ui| {
            self.render_status(ui);
        });
        egui::CentralPanel::default().show(ui, |ui| {
            // 必须在放内容之前记录可用范围：`Ui` 溢出时 max_rect 会跟着 min_rect 一起长。
            #[cfg(test)]
            {
                self.last_panel_rect = ui.max_rect();
            }
            match self.screen {
                Screen::Source => self.render_source(ui),
                Screen::Words => self.render_words(ui),
                Screen::Output => self.render_output(ui),
            }
            #[cfg(test)]
            {
                self.last_content_rect = ui.min_rect();
            }
        });

        self.render_exit_prompt(&ctx);
    }
}

impl eframe::App for DictionaryApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.draw(ui);
    }
}

fn segment_in_background(
    cached: Option<Arc<Jieba>>,
    options: &SegmentOptions,
    text: &str,
) -> SegmentOutcome {
    let jieba = match cached {
        Some(jieba) => jieba,
        None => {
            let mut jieba = Jieba::new();
            if let Some(path) = &options.user_dict_path {
                let result = File::open(path)
                    .map(BufReader::new)
                    .map_err(|error| format!("无法打开用户词典 {}：{error}", path.display()))
                    .and_then(|mut reader| {
                        jieba.load_dict(&mut reader).map_err(|error| {
                            format!("无法加载用户词典 {}：{error}", path.display())
                        })
                    });
                if let Err(error) = result {
                    return SegmentOutcome::Failed(error);
                }
            }
            Arc::new(jieba)
        }
    };

    let entries = extract_frequencies(
        &jieba,
        text,
        options.hmm,
        options.min_frequency,
        options.include_single_chars,
    );
    SegmentOutcome::Ready {
        jieba,
        entries,
        options: options.clone(),
    }
}

/// 截取前 `max_lines` 行用于预览，返回是否发生了截断。
fn preview_slice(text: &str, max_lines: usize) -> (&str, bool) {
    let mut end = 0;
    for (count, line) in text.split_inclusive('\n').enumerate() {
        if count == max_lines {
            return (&text[..end], true);
        }
        end += line.len();
    }
    (text, false)
}

fn install_font(context: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "sfss".to_owned(),
        Arc::new(egui::FontData::from_static(SFSS_FONT)),
    );
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "sfss".to_owned());
    fonts
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .insert(0, "sfss".to_owned());
    context.set_fonts(fonts);
}

fn configure_style(context: &egui::Context, theme: egui::Theme) {
    let mut style = (*context.style_of(theme)).clone();
    style.spacing.item_spacing = egui::vec2(10.0, 8.0);
    style.spacing.button_padding = egui::vec2(12.0, 7.0);
    style.text_styles.insert(
        egui::TextStyle::Heading,
        egui::FontId::new(24.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Body,
        egui::FontId::new(16.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Button,
        egui::FontId::new(16.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Monospace,
        egui::FontId::new(15.0, egui::FontFamily::Monospace),
    );
    context.set_style_of(theme, style);
}

fn editable_to_entry(row: &EditableWord) -> Result<WordEntry, String> {
    let word = row.word.trim();
    if word.is_empty() {
        return Err("词语不能为空".to_owned());
    }

    let frequency = if row.frequency.trim().is_empty() {
        None
    } else {
        Some(
            row.frequency
                .trim()
                .parse::<u64>()
                .map_err(|_| "词频必须是非负整数".to_owned())?,
        )
    };
    let pinyin: Vec<String> = row
        .pinyin
        .split(|character: char| {
            character.is_whitespace() || matches!(character, ',' | '，' | '/' | '、' | ';' | '；')
        })
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .collect();

    Ok(WordEntry {
        word: word.to_owned(),
        frequency,
        pinyin: (!pinyin.is_empty()).then_some(pinyin),
    })
}

fn render_editable_words(words: &[EditableWord]) -> String {
    let mut output = String::from("# 词语\t词频\t拼音（可选）\n");
    for row in words {
        if row.word.trim().is_empty() {
            continue;
        }
        output.push_str(row.word.trim());
        output.push('\t');
        output.push_str(row.frequency.trim());
        if !row.pinyin.trim().is_empty() {
            output.push('\t');
            output.push_str(row.pinyin.trim());
        }
        output.push('\n');
    }
    output
}

/// 读取文本文件。中文 Windows 上的 txt 常常是 GBK/ANSI，所以 UTF-8 之外还要回退解码。
fn read_text_file(path: &Path) -> Result<(String, &'static str), String> {
    let bytes =
        fs::read(path).map_err(|error| format!("无法读取文件 {}：{error}", path.display()))?;
    decode_text(&bytes).ok_or_else(|| {
        format!(
            "无法识别文件 {} 的编码，请另存为 UTF-8 后重试。",
            path.display()
        )
    })
}

fn decode_text(bytes: &[u8]) -> Option<(String, &'static str)> {
    if bytes.starts_with(&[0xFF, 0xFE]) || bytes.starts_with(&[0xFE, 0xFF]) {
        let encoding = if bytes[0] == 0xFF {
            encoding_rs::UTF_16LE
        } else {
            encoding_rs::UTF_16BE
        };
        let (text, had_errors) = encoding.decode_with_bom_removal(bytes);
        if !had_errors {
            return Some((text.into_owned(), encoding.name()));
        }
    }

    let without_bom = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes);
    if let Ok(text) = std::str::from_utf8(without_bom) {
        return Some((text.to_owned(), "UTF-8"));
    }

    // GB18030 是 GBK/GB2312 的超集，能覆盖简体中文 Windows 上的 ANSI 文本。
    for encoding in [encoding_rs::GB18030, encoding_rs::BIG5] {
        if let Some(text) = encoding.decode_without_bom_handling_and_without_replacement(bytes) {
            return Some((text.into_owned(), encoding.name()));
        }
    }

    None
}

fn write_utf8_bom_file(path: &Path, text: &str) -> Result<(), String> {
    let mut file =
        File::create(path).map_err(|error| format!("无法创建文件 {}：{error}", path.display()))?;
    file.write_all(b"\xEF\xBB\xBF")
        .and_then(|()| file.write_all(text.as_bytes()))
        .map_err(|error| format!("无法写入文件 {}：{error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 最小窗口尺寸，布局最紧的情况。
    const TEST_WINDOW: [f32; 2] = [960.0, 640.0];

    fn run_headless(screen: Screen, prepare: impl FnOnce(&mut DictionaryApp)) -> DictionaryApp {
        let context = egui::Context::default();
        let mut app = DictionaryApp::new(&context);
        app.screen = screen;
        prepare(&mut app);

        // 跑几帧让面板尺寸和表格的滚动状态稳定下来。
        for _ in 0..3 {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(TEST_WINDOW[0], TEST_WINDOW[1]),
                )),
                ..Default::default()
            };
            let _ = context.run_ui(input, |ui| app.draw(ui));
        }
        app
    }

    fn assert_content_fits(app: &DictionaryApp, what: &str) {
        assert!(
            app.last_content_rect.bottom() <= app.last_panel_rect.bottom() + 1.0,
            "{what} 溢出了面板：内容底边 {}，面板底边 {}",
            app.last_content_rect.bottom(),
            app.last_panel_rect.bottom()
        );
    }

    fn many_words(count: usize) -> Vec<WordEntry> {
        let pool: Vec<char> = ('\u{4e00}'..'\u{9fa5}')
            .filter(|character| flype_word::word_pinyin(&character.to_string()).is_ok())
            .take(200)
            .collect();
        (0..count)
            .map(|index| WordEntry {
                word: format!(
                    "{}{}",
                    pool[index % pool.len()],
                    pool[(index / pool.len()) % pool.len()]
                ),
                frequency: Some(index as u64),
                pinyin: None,
            })
            .collect()
    }

    #[test]
    fn source_page_keeps_the_start_button_inside_the_window() {
        let app = run_headless(Screen::Source, |app| {
            app.source_text = "这是一段用来撑爆文本框的中文原文。".repeat(12_000);
        });
        assert_content_fits(&app, "原文页（20 万字）");
    }

    #[test]
    fn output_page_keeps_the_preview_inside_the_window() {
        let app = run_headless(Screen::Output, |app| {
            app.replace_words(many_words(20_000));
        });
        assert!(app.conflicts.len() > 1, "这组测试数据应当产生重码");
        assert_content_fits(&app, "码表页（2 万词）");
    }

    #[test]
    fn words_page_keeps_the_table_inside_the_window() {
        let app = run_headless(Screen::Words, |app| {
            app.replace_words(many_words(20_000));
        });
        assert_content_fits(&app, "修订页（2 万行）");
    }

    #[test]
    fn theme_follows_the_toggle_not_the_operating_system() {
        let context = egui::Context::default();
        let _app = DictionaryApp::new(&context);

        // 系统切到深色时界面必须保持在按钮所指的浅色，否则开关状态就和界面对不上。
        let input = egui::RawInput {
            system_theme: Some(egui::Theme::Dark),
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(TEST_WINDOW[0], TEST_WINDOW[1]),
            )),
            ..Default::default()
        };
        let _ = context.run_ui(input, |_| {});
        assert!(
            !context.style_of(context.theme()).visuals.dark_mode,
            "系统主题变化把界面带跑了"
        );
    }

    #[test]
    fn rows_move_by_visible_order() {
        let context = egui::Context::default();
        let mut app = DictionaryApp::new(&context);
        app.replace_words(
            ["甲", "乙", "丙"]
                .into_iter()
                .map(|word| WordEntry {
                    word: word.to_owned(),
                    frequency: None,
                    pinyin: None,
                })
                .collect(),
        );

        // 跨过被筛选隐藏的“乙”，把“甲”移到“丙”下面。
        app.move_row(0, 2);
        let order: Vec<&str> = app.words.iter().map(|row| row.word.as_str()).collect();
        assert_eq!(order, ["乙", "丙", "甲"]);

        app.move_row(2, 0);
        let order: Vec<&str> = app.words.iter().map(|row| row.word.as_str()).collect();
        assert_eq!(order, ["甲", "乙", "丙"]);
    }

    #[test]
    fn importing_duplicates_reports_rows_instead_of_refusing_the_file() {
        let path = std::env::temp_dir().join("flype-word-duplicate-import.tsv");
        fs::write(&path, "小鹤\t3\n小鹤\t4\n词库\t2\n").unwrap();

        let context = egui::Context::default();
        let mut app = DictionaryApp::new(&context);
        app.load_word_list(&path);

        assert_eq!(app.words.len(), 3, "重复词语不该让整份文件被退回");
        assert_eq!(app.error_count(), 2, "两行重复都应标成错误");
        assert!(app.only_errors, "导入有问题时应自动切到“只看错误”");
        assert_eq!(app.words[2].code, "ciku", "没有重复的那一行应该正常出码");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn action_column_is_wide_enough_for_all_three_buttons() {
        // 断言的是表格真正用的那个常量，缩窄列宽就会让这个测试失败。
        let context = egui::Context::default();
        let _app = DictionaryApp::new(&context);
        let mut needed = 0.0;
        for _ in 0..2 {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(TEST_WINDOW[0], TEST_WINDOW[1]),
                )),
                ..Default::default()
            };
            let _ = context.run_ui(input, |ui| {
                ui.horizontal(|ui| {
                    let up = ui.add(egui::Button::new("↑").small());
                    let _down = ui.add(egui::Button::new("↓").small());
                    let delete = ui.small_button("删除");
                    needed = delete.rect.right() - up.rect.left();
                });
            });
        }
        assert!(
            needed <= ACTION_COLUMN_MIN_WIDTH,
            "操作列最小宽度 {ACTION_COLUMN_MIN_WIDTH} px 装不下三个按钮（需要 {needed} px）"
        );
    }

    #[test]
    fn editable_rows_support_manual_polyphonic_pinyin() {
        let row = EditableWord {
            word: "重庆".to_owned(),
            frequency: "12".to_owned(),
            pinyin: "chong qing".to_owned(),
            ..Default::default()
        };
        let encoded = encode_entries(&[editable_to_entry(&row).unwrap()]).unwrap();
        assert_eq!(encoded[0].code, "isqk");
        assert_eq!(encoded[0].frequency, Some(12));
    }

    #[test]
    fn editable_word_list_round_trips_through_core_parser() {
        let words = vec![
            EditableWord {
                word: "小鹤".to_owned(),
                frequency: "3".to_owned(),
                ..Default::default()
            },
            EditableWord {
                word: "音乐".to_owned(),
                pinyin: "yin yue".to_owned(),
                ..Default::default()
            },
        ];
        let parsed = flype_word::parse_word_list(&render_editable_words(&words)).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].frequency, Some(3));
        assert_eq!(
            parsed[1].pinyin.as_deref(),
            Some(&["yin".to_owned(), "yue".to_owned()][..])
        );
    }

    #[test]
    fn invalid_frequency_is_rejected() {
        let row = EditableWord {
            word: "词库".to_owned(),
            frequency: "很多".to_owned(),
            ..Default::default()
        };
        assert_eq!(editable_to_entry(&row).unwrap_err(), "词频必须是非负整数");
    }

    #[test]
    fn encoding_cache_only_recomputes_after_edits() {
        let mut row = EditableWord {
            word: "小鹤".to_owned(),
            ..Default::default()
        };
        row.refresh_encoding();
        assert_eq!(row.cached.as_ref().unwrap().as_ref().unwrap().code, "xnhe");

        // 未修改时缓存必须原样保留（这里塞一个假值来验证没有重算）。
        row.cached = Some(Ok(EncodedEntry {
            word: "小鹤".to_owned(),
            code: "cached".to_owned(),
            frequency: None,
        }));
        row.refresh_encoding();
        assert_eq!(
            row.cached.as_ref().unwrap().as_ref().unwrap().code,
            "cached"
        );

        // 改了拼音就必须重算。
        row.pinyin = "xiao he".to_owned();
        row.refresh_encoding();
        assert_eq!(row.cached.as_ref().unwrap().as_ref().unwrap().code, "xnhe");
    }

    #[test]
    fn gbk_and_utf8_files_are_both_decoded() {
        let (text, encoding) = decode_text("中文\n".as_bytes()).unwrap();
        assert_eq!(text, "中文\n");
        assert_eq!(encoding, "UTF-8");

        let (text, encoding) = decode_text(&[0xD6, 0xD0, 0xCE, 0xC4, 0x0D, 0x0A]).unwrap();
        assert_eq!(text, "中文\r\n");
        assert_eq!(encoding, "gb18030");

        let (text, encoding) = decode_text(&[0xEF, 0xBB, 0xBF, 0xE4, 0xB8, 0xAD]).unwrap();
        assert_eq!(text, "中");
        assert_eq!(encoding, "UTF-8");
    }

    #[test]
    fn preview_is_truncated_by_line_count() {
        let text = "a\nb\nc\nd\n";
        assert_eq!(preview_slice(text, 2), ("a\nb\n", true));
        assert_eq!(preview_slice(text, 10), (text, false));
        assert_eq!(preview_slice("", 10), ("", false));
    }
}
