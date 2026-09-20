import os
import time
import paramiko

PASSWORD = os.environ["AUTODL_PASSWORD"]
REMOTE = "/root/autodl-tmp/FT-CNN"


def main():
    for i in range(40):  # ~40 * 60s = 40 min max
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(
            "connect.nmb2.seetacloud.com",
            28950,
            username="root",
            password=PASSWORD,
            timeout=30,
        )
        _, o, _ = c.exec_command(
            "ps -ef | grep 'src.train' | grep -v grep || echo NO_TRAIN; "
            f"tail -n 3 {REMOTE}/logs/retrain_pipeline.log 2>/dev/null; "
            f"grep -E 'ALL_DONE|Accuracy:|use_class_weight' {REMOTE}/logs/retrain_eval.log {REMOTE}/logs/retrain.log {REMOTE}/logs/retrain_pipeline.log 2>/dev/null | tail -n 20",
            timeout=30,
        )
        text = o.read().decode("utf-8", "replace")
        print(f"=== poll {i} ===")
        print(text[-1500:].encode("ascii", "replace").decode())
        if "ALL_DONE" in text or (
            "NO_TRAIN" in text and "Accuracy:" in text
        ):
            _, m, _ = c.exec_command(f"cat {REMOTE}/artifacts/metrics.json", timeout=30)
            metrics = m.read().decode("utf-8", "replace")
            open(r"C:\Users\安\Desktop\ECG\FT-CNN\artifacts\metrics.json", "w", encoding="utf-8").write(metrics)
            print("METRICS_SAVED")
            print(metrics[:2500])
            c.close()
            return
        c.close()
        time.sleep(60)
    print("TIMEOUT_WAITING")


if __name__ == "__main__":
    main()
