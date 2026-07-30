use flype_word::{
    encode_entries, extract_frequencies, find_conflicts, parse_word_list, render_conflicts,
    render_encoded, render_frequency_tsv,
};
use jieba_rs::Jieba;
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const HELP: &str = "\
flype-word - 小鹤双拼词库生成器

用法：
  flype-word extract <原始文本> <词频表> [选项]
  flype-word encode  <修订词表> <编码词库> [选项]

extract 选项：
  --min-frequency <次数>  最低词频，默认 1
  --include-single        保留单字，默认只导出二字及以上词语
  --user-dict <文件>      加载 jieba 用户词典（词语 词频 词性）
  --no-hmm                关闭 jieba 的 HMM 新词识别

encode 选项：
  --conflicts <文件>      另存重码报告
  --with-flypy-header     添加小鹤用户码表的配置头

输入或输出路径写成 - 时，表示标准输入或标准输出。
使用 flype-word <命令> --help 查看命令示例。
";

const EXTRACT_HELP: &str = "\
第一步：分词并统计词频

  flype-word extract article.txt words.tsv
  flype-word extract article.txt words.tsv --min-frequency 2 --include-single
  flype-word extract - words.tsv --user-dict jieba-user.txt

输出 words.tsv 后可人工删词、加词或调整顺序。多音字可在第三列填写拼音：
  重庆    20    chong qing
";

const ENCODE_HELP: &str = "\
第三步：为人工修订后的词表编码

  flype-word encode words.tsv flypy-user.txt
  flype-word encode words.tsv flypy-user.txt --conflicts conflicts.tsv
  flype-word encode words.tsv flypy-user.txt --with-flypy-header

编码规则：
  单字：完整双拼
  二字：每字完整双拼，共四码
  三字：前两字拼音首字母 + 第三字完整双拼
  四字及以上：前三字和末字的拼音首字母
";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("错误：{error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<String> = env::args().collect();
    let Some(command) = arguments.get(1).map(String::as_str) else {
        print!("{HELP}");
        return Ok(());
    };

    match command {
        "-h" | "--help" | "help" => {
            print!("{HELP}");
            Ok(())
        }
        "-V" | "--version" | "version" => {
            println!("flype-word {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        "extract" => run_extract(&arguments[2..]),
        "encode" => run_encode(&arguments[2..]),
        unknown => Err(format!("未知命令“{unknown}”。运行 flype-word --help 查看用法").into()),
    }
}

fn run_extract(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    if arguments
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        print!("{EXTRACT_HELP}");
        return Ok(());
    }

    let mut positional = Vec::new();
    let mut min_frequency = 1;
    let mut include_single = false;
    let mut hmm = true;
    let mut user_dict = None;
    let mut index = 0;

    while index < arguments.len() {
        match arguments[index].as_str() {
            "--min-frequency" => {
                let value = option_value(arguments, index, "--min-frequency")?;
                min_frequency = value
                    .parse::<u64>()
                    .map_err(|_| "--min-frequency 必须是非负整数")?;
                index += 2;
            }
            "--include-single" => {
                include_single = true;
                index += 1;
            }
            "--no-hmm" => {
                hmm = false;
                index += 1;
            }
            "--user-dict" => {
                user_dict = Some(PathBuf::from(option_value(
                    arguments,
                    index,
                    "--user-dict",
                )?));
                index += 2;
            }
            value if value.starts_with('-') && value != "-" => {
                return Err(format!("extract 不支持选项“{value}”").into());
            }
            value => {
                positional.push(value.to_owned());
                index += 1;
            }
        }
    }

    let (input_path, output_path) = two_paths(&positional, "extract")?;
    let text = read_text(input_path)?;

    let mut jieba = Jieba::new();
    if let Some(path) = user_dict {
        let mut reader = BufReader::new(
            File::open(&path)
                .map_err(|error| format!("无法打开 jieba 用户词典 {}：{error}", path.display()))?,
        );
        jieba
            .load_dict(&mut reader)
            .map_err(|error| format!("无法加载 jieba 用户词典 {}：{error}", path.display()))?;
    }

    let entries = extract_frequencies(&jieba, &text, hmm, min_frequency, include_single);
    let count = entries.len();
    write_text(output_path, &render_frequency_tsv(&entries))?;
    eprintln!("已导出 {count} 个词语。");
    Ok(())
}

fn run_encode(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    if arguments
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        print!("{ENCODE_HELP}");
        return Ok(());
    }

    let mut positional = Vec::new();
    let mut conflicts_path = None;
    let mut with_flypy_header = false;
    let mut index = 0;

    while index < arguments.len() {
        match arguments[index].as_str() {
            "--conflicts" => {
                conflicts_path = Some(PathBuf::from(option_value(
                    arguments,
                    index,
                    "--conflicts",
                )?));
                index += 2;
            }
            "--with-flypy-header" => {
                with_flypy_header = true;
                index += 1;
            }
            value if value.starts_with('-') && value != "-" => {
                return Err(format!("encode 不支持选项“{value}”").into());
            }
            value => {
                positional.push(value.to_owned());
                index += 1;
            }
        }
    }

    let (input_path, output_path) = two_paths(&positional, "encode")?;
    let input = read_text(input_path)?;
    let word_entries = parse_word_list(&input)?;
    let encoded = encode_entries(&word_entries)?;
    let conflicts = find_conflicts(&encoded);

    write_text(output_path, &render_encoded(&encoded, with_flypy_header))?;
    if let Some(path) = conflicts_path {
        write_text(&path, &render_conflicts(&conflicts))?;
    }

    eprintln!(
        "已编码 {} 个词语，发现 {} 组重码。",
        encoded.len(),
        conflicts.len()
    );
    Ok(())
}

fn option_value<'a>(
    arguments: &'a [String],
    index: usize,
    option: &str,
) -> Result<&'a str, Box<dyn Error>> {
    arguments
        .get(index + 1)
        .map(String::as_str)
        .ok_or_else(|| format!("选项 {option} 缺少参数").into())
}

fn two_paths<'a>(
    positional: &'a [String],
    command: &str,
) -> Result<(&'a Path, &'a Path), Box<dyn Error>> {
    if positional.len() != 2 {
        return Err(format!(
            "{command} 需要两个路径参数；运行 flype-word {command} --help 查看示例"
        )
        .into());
    }
    Ok((Path::new(&positional[0]), Path::new(&positional[1])))
}

fn read_text(path: &Path) -> Result<String, Box<dyn Error>> {
    let mut text = String::new();
    if path == Path::new("-") {
        io::stdin().read_to_string(&mut text)?;
    } else {
        File::open(path)
            .map_err(|error| format!("无法打开输入文件 {}：{error}", path.display()))?
            .read_to_string(&mut text)
            .map_err(|error| format!("无法读取 UTF-8 输入文件 {}：{error}", path.display()))?;
    }
    Ok(text)
}

fn write_text(path: &Path, text: &str) -> Result<(), Box<dyn Error>> {
    if path == Path::new("-") {
        let mut output = io::stdout().lock();
        output.write_all(text.as_bytes())?;
        output.flush()?;
    } else {
        let mut output = File::create(path)
            .map_err(|error| format!("无法创建输出文件 {}：{error}", path.display()))?;
        output
            .write_all(b"\xEF\xBB\xBF")
            .map_err(|error| format!("无法写入输出文件 {}：{error}", path.display()))?;
        output
            .write_all(text.as_bytes())
            .map_err(|error| format!("无法写入输出文件 {}：{error}", path.display()))?;
    }
    Ok(())
}
