use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new(name: &str) -> Self {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("系统时间应晚于 Unix epoch")
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("flype-word-{name}-{}-{unique}", std::process::id()));
        fs::create_dir_all(&path).expect("应能创建临时测试目录");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_flype-word")
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn run_success(command: &mut Command) -> std::process::Output {
    let output = command.output().expect("命令应能启动");
    assert!(
        output.status.success(),
        "命令执行失败\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn read_utf8_bom_file(path: &Path) -> String {
    let bytes = fs::read(path).expect("应能读取输出文件");
    assert!(
        bytes.starts_with(&[0xef, 0xbb, 0xbf]),
        "{} 应包含 UTF-8 BOM",
        path.display()
    );
    String::from_utf8(bytes[3..].to_vec()).expect("BOM 后应为有效 UTF-8")
}

#[test]
fn real_text_extract_then_encode_end_to_end() {
    let temporary = TestDirectory::new("e2e");
    let frequencies = temporary.path().join("words.tsv");
    let dictionary = temporary.path().join("flypy-user.txt");
    let conflicts = temporary.path().join("conflicts.tsv");

    let extract_output = run_success(
        Command::new(binary())
            .arg("extract")
            .arg(fixture("article.txt"))
            .arg(&frequencies)
            .arg("--user-dict")
            .arg(fixture("jieba-user.txt"))
            .arg("--min-frequency")
            .arg("2"),
    );
    let extract_stderr = String::from_utf8_lossy(&extract_output.stderr);
    assert!(extract_stderr.contains("已导出"));

    let frequency_text = read_utf8_bom_file(&frequencies);
    assert!(
        frequency_text.lines().any(|line| line == "小鹤双拼\t3"),
        "实际词频表：\n{frequency_text}"
    );
    assert!(
        frequency_text.lines().any(|line| line == "词库生成器\t2"),
        "实际词频表：\n{frequency_text}"
    );

    let encode_output = run_success(
        Command::new(binary())
            .arg("encode")
            .arg(fixture("revised-words.tsv"))
            .arg(&dictionary)
            .arg("--conflicts")
            .arg(&conflicts)
            .arg("--with-flypy-header"),
    );
    assert_eq!(
        String::from_utf8_lossy(&encode_output.stderr),
        "已编码 9 个词语，发现 1 组重码。\n"
    );

    let dictionary_text = read_utf8_bom_file(&dictionary);
    assert_eq!(
        dictionary_text,
        "\
---config@码表分类=主码-用户码表
---config@码表别名=用户
小鹤\txnhe
计算机\tjsji
知识库\tvuku
双拼方案\tupfa
中华人民共和国\tvhrg
重庆\tisqk
音乐\tybyt
时\tui
是\tui
"
    );

    let conflict_text = read_utf8_bom_file(&conflicts);
    assert_eq!(conflict_text, "# 编码\t重码词\nui\t时 是\n");
}

#[test]
fn stdin_stdout_pipeline_is_clean_utf8_without_bom() {
    let mut child = Command::new(binary())
        .args(["encode", "-", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("命令应能启动");

    child
        .stdin
        .take()
        .expect("stdin 应存在")
        .write_all("小鹤\n".as_bytes())
        .expect("应能写入 stdin");
    let output = child.wait_with_output().expect("命令应能结束");

    assert!(output.status.success());
    assert_eq!(output.stdout, "小鹤\txnhe\n".as_bytes());
    assert!(!output.stdout.starts_with(&[0xef, 0xbb, 0xbf]));
    assert_eq!(
        String::from_utf8_lossy(&output.stderr),
        "已编码 1 个词语，发现 0 组重码。\n"
    );
}

#[test]
fn invalid_manual_pronunciation_fails_without_creating_dictionary() {
    let temporary = TestDirectory::new("invalid");
    let dictionary = temporary.path().join("should-not-exist.txt");

    let output = Command::new(binary())
        .arg("encode")
        .arg(fixture("invalid-pronunciation.tsv"))
        .arg(&dictionary)
        .output()
        .expect("命令应能启动");

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("词语“重庆”有 2 个字，但人工拼音有 1 个音节")
    );
    assert!(!dictionary.exists(), "校验失败时不应留下不完整的词库文件");
}
