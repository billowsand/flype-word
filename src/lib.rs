use jieba_rs::Jieba;
use pinyin::ToPinyin;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrequencyEntry {
    pub word: String,
    pub frequency: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordEntry {
    pub word: String,
    pub frequency: Option<u64>,
    pub pinyin: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedEntry {
    pub word: String,
    pub code: String,
    pub frequency: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlypeError(String);

impl FlypeError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for FlypeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for FlypeError {}

/// 使用 jieba 精确模式分词，只保留 pinyin 数据库能够识别的汉字词。
pub fn extract_frequencies(
    jieba: &Jieba,
    text: &str,
    hmm: bool,
    min_frequency: u64,
    include_single_chars: bool,
) -> Vec<FrequencyEntry> {
    let mut frequencies = HashMap::<String, u64>::new();

    for token in jieba.cut(text, hmm) {
        let word = token.word.trim();
        let char_count = word.chars().count();
        if word.is_empty()
            || (!include_single_chars && char_count == 1)
            || !word
                .chars()
                .all(|character| character.to_pinyin().is_some())
        {
            continue;
        }
        *frequencies.entry(word.to_owned()).or_default() += 1;
    }

    let mut entries: Vec<_> = frequencies
        .into_iter()
        .filter(|(_, frequency)| *frequency >= min_frequency)
        .map(|(word, frequency)| FrequencyEntry { word, frequency })
        .collect();

    entries.sort_by(|left, right| {
        right
            .frequency
            .cmp(&left.frequency)
            .then_with(|| left.word.cmp(&right.word))
    });
    entries
}

pub fn render_frequency_tsv(entries: &[FrequencyEntry]) -> String {
    let mut output = String::from("# 词语\t词频\t拼音（可选；多音字可填空格分隔的无声调拼音）\n");
    for entry in entries {
        output.push_str(&entry.word);
        output.push('\t');
        output.push_str(&entry.frequency.to_string());
        output.push('\n');
    }
    output
}

/// 读取人工修订后的词表，重复词语视为错误。
///
/// 支持以下形式：
/// - `词语`
/// - `词语<TAB>词频`
/// - `词语<TAB>词频<TAB>pin yin`
/// - `词语<TAB>pin yin`
pub fn parse_word_list(input: &str) -> Result<Vec<WordEntry>, FlypeError> {
    let mut seen = HashSet::new();
    let mut entries = Vec::new();

    for (line_number, entry) in parse_word_lines(input) {
        if !seen.insert(entry.word.clone()) {
            return Err(FlypeError::new(format!(
                "第 {line_number} 行出现重复词语“{}”",
                entry.word
            )));
        }
        entries.push(entry);
    }

    Ok(entries)
}

/// 与 [`parse_word_list`] 相同，但保留重复词语，交给调用方逐行提示。
///
/// 图形界面用它来导入词表：重复词语会显示成行错误，方便当场修改，
/// 而不是整份文件拒绝导入。
pub fn parse_word_list_lenient(input: &str) -> Vec<WordEntry> {
    parse_word_lines(input)
        .into_iter()
        .map(|(_, entry)| entry)
        .collect()
}

/// 逐行解析词表，返回行号和词条，不做任何跨行校验。
fn parse_word_lines(input: &str) -> Vec<(usize, WordEntry)> {
    let mut entries = Vec::new();

    for (line_index, original_line) in input.lines().enumerate() {
        let line_number = line_index + 1;
        let line = original_line.trim().trim_start_matches('\u{feff}');
        if line.is_empty() || line.starts_with('#') || line.starts_with("---config@") {
            continue;
        }

        let fields: Vec<&str> = if line.contains('\t') {
            line.split('\t').map(str::trim).collect()
        } else {
            line.split_whitespace().collect()
        };
        let word = fields.first().copied().unwrap_or_default();
        if word.is_empty() {
            continue;
        }

        let mut frequency = None;
        let mut pinyin_start = 1;
        if let Some(value) = fields.get(1).filter(|value| !value.is_empty())
            && let Ok(parsed) = value.parse::<u64>()
        {
            frequency = Some(parsed);
            pinyin_start = 2;
        }

        let pinyin_text = fields
            .get(pinyin_start..)
            .unwrap_or_default()
            .iter()
            .filter(|value| !value.is_empty())
            .copied()
            .collect::<Vec<_>>()
            .join(" ");
        let pronunciation: Vec<String> = pinyin_text
            .split(|character: char| {
                character.is_whitespace()
                    || matches!(character, ',' | '，' | '/' | '、' | ';' | '；')
            })
            .filter(|value| !value.is_empty())
            .map(normalize_syllable)
            .collect();

        entries.push((
            line_number,
            WordEntry {
                word: word.to_owned(),
                frequency,
                pinyin: (!pronunciation.is_empty()).then_some(pronunciation),
            },
        ));
    }

    entries
}

pub fn encode_entries(entries: &[WordEntry]) -> Result<Vec<EncodedEntry>, FlypeError> {
    entries
        .iter()
        .map(|entry| {
            let syllables = match &entry.pinyin {
                Some(syllables) => {
                    let char_count = entry.word.chars().count();
                    if syllables.len() != char_count {
                        return Err(FlypeError::new(format!(
                            "词语“{}”有 {char_count} 个字，但人工拼音有 {} 个音节",
                            entry.word,
                            syllables.len()
                        )));
                    }
                    syllables.clone()
                }
                None => word_pinyin(&entry.word)?,
            };

            Ok(EncodedEntry {
                word: entry.word.clone(),
                code: encode_word_from_pinyin(&syllables)?,
                frequency: entry.frequency,
            })
        })
        .collect()
}

pub fn render_encoded(entries: &[EncodedEntry], with_flypy_header: bool) -> String {
    let mut output = String::new();
    if with_flypy_header {
        output.push_str("---config@码表分类=主码-用户码表\n");
        output.push_str("---config@码表别名=用户\n");
    }
    for entry in entries {
        output.push_str(&entry.word);
        output.push('\t');
        output.push_str(&entry.code);
        output.push('\n');
    }
    output
}

pub fn find_conflicts(entries: &[EncodedEntry]) -> BTreeMap<String, Vec<String>> {
    let mut by_code = BTreeMap::<String, Vec<String>>::new();
    for entry in entries {
        by_code
            .entry(entry.code.clone())
            .or_default()
            .push(entry.word.clone());
    }
    by_code.retain(|_, words| words.len() > 1);
    by_code
}

pub fn render_conflicts(conflicts: &BTreeMap<String, Vec<String>>) -> String {
    let mut output = String::from("# 编码\t重码词\n");
    for (code, words) in conflicts {
        output.push_str(code);
        output.push('\t');
        output.push_str(&words.join(" "));
        output.push('\n');
    }
    output
}

pub fn word_pinyin(word: &str) -> Result<Vec<String>, FlypeError> {
    word.chars()
        .map(|character| {
            character
                .to_pinyin()
                .map(|value| normalize_syllable(value.plain()))
                .ok_or_else(|| {
                    FlypeError::new(format!("词语“{word}”中的字符“{character}”无法转换为拼音"))
                })
        })
        .collect()
}

pub fn encode_word_from_pinyin(syllables: &[String]) -> Result<String, FlypeError> {
    match syllables {
        [] => Err(FlypeError::new("不能编码空词语")),
        [single] => encode_syllable(single),
        [first, second] => Ok(format!(
            "{}{}",
            encode_syllable(first)?,
            encode_syllable(second)?
        )),
        [first, second, third] => Ok(format!(
            "{}{}{}",
            first_letter(first)?,
            first_letter(second)?,
            encode_syllable(third)?
        )),
        many => Ok(format!(
            "{}{}{}{}",
            first_letter(&many[0])?,
            first_letter(&many[1])?,
            first_letter(&many[2])?,
            first_letter(many.last().expect("已确认词语非空"))?
        )),
    }
}

pub fn encode_syllable(raw_syllable: &str) -> Result<String, FlypeError> {
    let syllable = normalize_syllable(raw_syllable);
    if let Some(code) = zero_initial_code(&syllable) {
        return Ok(code);
    }

    let (initial_key, final_part) = split_syllable(&syllable)
        .ok_or_else(|| FlypeError::new(format!("无法识别拼音音节“{raw_syllable}”的声母")))?;
    let final_key = final_key(final_part).ok_or_else(|| {
        FlypeError::new(format!(
            "无法识别拼音音节“{raw_syllable}”的韵母“{final_part}”"
        ))
    })?;

    Ok(format!("{initial_key}{final_key}"))
}

/// 零声母音节（拼音以韵母开头）的打法：
///
/// - 单字母韵母：重复两次该键，`啊 a → aa`、`哦 o → oo`、`额 e → ee`；
/// - 双字母韵母：直接打全拼，`爱 ai → ai`、`恩 en → en`、`欧 ou → ou`、`儿 er → er`；
/// - 三字母韵母：韵母首字母 + 韵母键，`昂 ang → ah`、`鞥 eng → eg`。
///
/// 不是零声母音节（或不是合法的零声母韵母）时返回 `None`。
fn zero_initial_code(syllable: &str) -> Option<String> {
    const ZERO_INITIAL_FINALS: [&str; 12] = [
        "a", "o", "e", "ai", "ei", "ao", "ou", "an", "en", "er", "ang", "eng",
    ];
    if !ZERO_INITIAL_FINALS.contains(&syllable) {
        return None;
    }

    let first = syllable.chars().next()?;
    match syllable.len() {
        1 => Some(format!("{first}{first}")),
        2 => Some(syllable.to_owned()),
        _ => final_key(syllable).map(|key| format!("{first}{key}")),
    }
}

/// 韵母在小鹤双拼键盘上的键位。
fn final_key(final_part: &str) -> Option<char> {
    let normalized_final = match final_part {
        "iou" => "iu",
        "uei" => "ui",
        "uen" => "un",
        "üe" => "ve",
        value => value,
    };
    Some(match normalized_final {
        "a" => 'a',
        "o" => 'o',
        "e" => 'e',
        "i" => 'i',
        "u" => 'u',
        "v" | "ü" => 'v',
        "ai" => 'd',
        "ei" => 'w',
        "ui" => 'v',
        "ao" => 'c',
        "ou" => 'z',
        "iu" => 'q',
        "ie" => 'p',
        "ue" | "ve" => 't',
        "er" => 'r',
        "an" => 'j',
        "en" => 'f',
        "in" => 'b',
        "un" | "vn" => 'y',
        "ang" => 'h',
        "eng" => 'g',
        "ing" => 'k',
        "ong" | "iong" => 's',
        "ia" | "ua" => 'x',
        "ian" => 'm',
        "uan" => 'r',
        "iang" | "uang" => 'l',
        "iao" => 'n',
        "uai" => 'k',
        "uo" => 'o',
        _ => return None,
    })
}

/// 拆出音节的小鹤声母键和剩下的韵母。
///
/// 小鹤双拼里 `zh`/`ch`/`sh` 打在 `v`/`i`/`u` 上，所以声母键不一定等于拼音首字母。
/// 零声母音节（`an`、`ou` 等）没有声母，返回 `None`。
fn split_syllable(syllable: &str) -> Option<(char, &str)> {
    for (initial, key) in [("zh", 'v'), ("ch", 'i'), ("sh", 'u')] {
        if let Some(final_part) = syllable.strip_prefix(initial) {
            return Some((key, final_part));
        }
    }
    let first = syllable.chars().next()?;
    if "bpmfdtnlgkhjqxrzcsyw".contains(first) {
        Some((first, &syllable[first.len_utf8()..]))
    } else {
        None
    }
}

/// 词组编码里“取首字母”用的那一键。
///
/// 必须和全码的第一键一致：`知/识` 的全码是 `vi`/`ui`，所以 `知识库` 是 `vuku`
/// 而不是按拼音首字母拼出来的 `zsku`。零声母音节取韵母的第一个字母，
/// 这本来就等于它全码的第一键。
fn first_letter(raw_syllable: &str) -> Result<char, FlypeError> {
    let syllable = normalize_syllable(raw_syllable);
    if let Some((key, _)) = split_syllable(&syllable) {
        return Ok(key);
    }
    syllable
        .chars()
        .next()
        .filter(|character| character.is_ascii_lowercase())
        .ok_or_else(|| FlypeError::new(format!("无法取得拼音“{raw_syllable}”的首字母")))
}

fn normalize_syllable(raw_syllable: &str) -> String {
    raw_syllable
        .trim()
        .to_lowercase()
        .replace("u:", "v")
        .replace('ü', "v")
        .trim_end_matches(|character: char| ('0'..='5').contains(&character))
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn syllables(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn encodes_xiaohe_syllables() {
        let cases = [
            ("xiao", "xn"),
            ("he", "he"),
            ("shuang", "ul"),
            ("pin", "pb"),
            ("yin", "yb"),
            ("zhong", "vs"),
            ("lüe", "lt"),
            ("ju", "ju"),
            ("yuan", "yr"),
            ("ang", "ah"),
        ];
        for (pinyin, expected) in cases {
            assert_eq!(encode_syllable(pinyin).unwrap(), expected);
        }
    }

    #[test]
    fn applies_word_length_rules() {
        assert_eq!(
            encode_word_from_pinyin(&syllables(&["xiao", "he"])).unwrap(),
            "xnhe"
        );
        assert_eq!(
            encode_word_from_pinyin(&syllables(&["ji", "suan", "ji"])).unwrap(),
            "jsji"
        );
        assert_eq!(
            encode_word_from_pinyin(&syllables(&["shuang", "pin", "fang", "an"])).unwrap(),
            "upfa"
        );
        assert_eq!(
            encode_word_from_pinyin(&syllables(&[
                "zhong", "hua", "ren", "min", "gong", "he", "guo"
            ]))
            .unwrap(),
            "vhrg"
        );
    }

    #[test]
    fn zero_initial_syllables_follow_the_length_rules() {
        // 单字母韵母重复两次
        for (syllable, expected) in [("a", "aa"), ("o", "oo"), ("e", "ee")] {
            assert_eq!(encode_syllable(syllable).unwrap(), expected);
        }
        // 双字母韵母直接打全拼
        for syllable in ["ai", "ei", "ao", "ou", "an", "en", "er"] {
            assert_eq!(encode_syllable(syllable).unwrap(), syllable);
        }
        // 三字母韵母取首字母 + 韵母键
        assert_eq!(encode_syllable("ang").unwrap(), "ah");
        assert_eq!(encode_syllable("eng").unwrap(), "eg");

        // 有声母时，同样的韵母走键位表：海 hai → hd，而零声母的 爱 ai → ai。
        assert_eq!(encode_syllable("hai").unwrap(), "hd");
        assert_eq!(encode_syllable("gei").unwrap(), "gw");

        // 不是合法零声母韵母的输入仍然要报错，不能原样当成编码。
        assert!(encode_syllable("ez").is_err());
        assert!(encode_syllable("aq").is_err());
    }

    #[test]
    fn word_codes_use_the_xiaohe_initial_key_for_zh_ch_sh() {
        // 取首字母要取小鹤的声母键（zh/ch/sh → v/i/u），不是拼音首字母。
        assert_eq!(
            encode_word_from_pinyin(&syllables(&["zhi", "shi", "ku"])).unwrap(),
            "vuku"
        );
        assert_eq!(
            encode_word_from_pinyin(&syllables(&["chi", "fan"])).unwrap(),
            "iifj"
        );
        assert_eq!(
            encode_word_from_pinyin(&syllables(&["chong", "qing", "shi"])).unwrap(),
            "iqui"
        );
        assert_eq!(
            encode_word_from_pinyin(&syllables(&["zhong", "guo", "ren", "min"])).unwrap(),
            "vgrm"
        );
        // 首字母必须和该音节全码的第一键一致。
        for syllable in [
            "zhi", "chi", "shi", "zha", "chuang", "shuo", "si", "ci", "zi",
        ] {
            let full = encode_syllable(syllable).unwrap();
            assert_eq!(
                first_letter(syllable).unwrap(),
                full.chars().next().unwrap(),
                "音节 {syllable} 的首字母和全码 {full} 的第一键不一致"
            );
        }
    }

    #[test]
    fn parses_optional_frequency_and_pinyin() {
        let entries =
            parse_word_list("# comment\n小鹤双拼\t12\n重庆\t9\tchong qing\n音乐\tyin yue\n词语\n")
                .unwrap();
        assert_eq!(entries.len(), 4);
        assert_eq!(entries[0].frequency, Some(12));
        assert_eq!(
            entries[1].pinyin.as_deref(),
            Some(&syllables(&["chong", "qing"])[..])
        );
        assert_eq!(entries[2].frequency, None);
        assert_eq!(
            entries[2].pinyin.as_deref(),
            Some(&syllables(&["yin", "yue"])[..])
        );
        assert_eq!(entries[3].pinyin, None);
    }

    #[test]
    fn strict_parsing_rejects_duplicates_but_lenient_keeps_them() {
        let input = "小鹤\t3\n小鹤\t4\n";
        assert!(parse_word_list(input).is_err());
        let entries = parse_word_list_lenient(input);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].word, "小鹤");
        assert_eq!(entries[1].frequency, Some(4));
    }

    #[test]
    fn rejects_wrong_pronunciation_length() {
        let entries = vec![WordEntry {
            word: "重庆".to_owned(),
            frequency: None,
            pinyin: Some(syllables(&["chong"])),
        }];
        assert!(encode_entries(&entries).is_err());
    }

    #[test]
    fn reports_code_conflicts() {
        let entries = vec![
            EncodedEntry {
                word: "甲乙".to_owned(),
                code: "abcd".to_owned(),
                frequency: None,
            },
            EncodedEntry {
                word: "丙丁".to_owned(),
                code: "abcd".to_owned(),
                frequency: None,
            },
        ];
        let conflicts = find_conflicts(&entries);
        assert_eq!(conflicts["abcd"], ["甲乙", "丙丁"]);
    }
}
