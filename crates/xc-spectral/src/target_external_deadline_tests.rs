use super::*;

#[test]
fn pipe_child() {
    if std::env::var_os("XC_R2_PIPE_CHILD").is_some() {
        std::thread::sleep(Duration::from_secs(30));
    }
}

#[test]
fn request_write_and_reply_share_one_deadline() {
    let executable = std::env::current_exe().unwrap();
    // The child deliberately never reads stdin. The larger frames exceed both
    // Windows and Linux pipe capacity; the tiny frame exercises reply timeout.
    for length in [60, 4 * 1024 * 1024, 7 * 1024 * 1024] {
        let mut command = Command::new(&executable);
        command
            .args(["--exact", "target::external::deadline_tests::pipe_child"])
            .env("XC_R2_PIPE_CHILD", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000);
        }
        let mut child = command.spawn().unwrap();
        let (requests, writes) = request_writer(BufWriter::new(child.stdin.take().unwrap()));
        let (_reply_sender, replies) = mpsc::sync_channel(1);
        let mut connection = Connection {
            child,
            requests,
            writes,
            replies,
            timeout: Duration::from_millis(200),
            failed: false,
            next_request_id: 1,
            precision_bits: 128,
            _executable_file: File::open(&executable).unwrap(),
        };
        let message = serde_json::json!({"operation":"initialize", "input":"x".repeat(length)});
        let start = Instant::now();
        let error = connection.exchange(&message).unwrap_err();
        assert!(error.to_string().contains("timed out"), "{error:#}");
        assert!(
            start.elapsed() < Duration::from_secs(3),
            "frame length={length}"
        );
        assert!(connection.failed);
        // A timed-out request cannot be resumed or mistaken for the next reply.
        assert!(connection
            .exchange(&message)
            .unwrap_err()
            .to_string()
            .contains("earlier failure"));
        connection.child.wait().unwrap();
    }
}
