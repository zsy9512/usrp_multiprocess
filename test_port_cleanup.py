#!/usr/bin/env python3
"""
测试端口清理功能
"""

import subprocess
import time
import socket
import sys

def test_port_cleanup():
    """测试端口清理功能"""
    print("🧪 测试端口清理功能")
    print("=" * 50)

    # 首先检查端口是否被占用
    def check_port(host, port):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0  # 0表示连接成功，端口被占用
        except:
            return False

    host = "127.0.0.1"
    port = 50000

    print(f"🔍 检查端口 {host}:{port} 状态...")
    is_occupied = check_port(host, port)

    if is_occupied:
        print("⚠️  端口已被占用")
        print("🔧 启动队列服务器并使用 --force 参数清理端口...")

        # 启动服务器，使用 --force 参数
        try:
            process = subprocess.Popen([
                sys.executable, "queue_server.py",
                "--host", host,
                "--port", str(port),
                "--force"
            ])

            print(f"✅ 服务器进程已启动 (PID: {process.pid})")

            # 等待服务器启动或失败
            time.sleep(5)

            # 检查进程是否还在运行
            if process.poll() is None:
                print("✅ 服务器成功启动")

                # 发送SIGINT信号停止服务器
                try:
                    process.terminate()
                    time.sleep(2)

                    if process.poll() is None:
                        process.kill()

                    process.wait(timeout=5)
                    print("✅ 服务器已停止")
                except Exception as e:
                    print(f"❌ 停止服务器时出错: {e}")
            else:
                print(f"❌ 服务器启动失败，退出码: {process.returncode}")

        except Exception as e:
            print(f"❌ 启动服务器时出错: {e}")

    else:
        print("✅ 端口可用，无需清理")
        print("🔧 启动队列服务器进行测试...")

        # 启动服务器
        try:
            process = subprocess.Popen([
                sys.executable, "queue_server.py",
                "--host", host,
                "--port", str(port)
            ])

            print(f"✅ 服务器进程已启动 (PID: {process.pid})")

            # 等待服务器启动
            time.sleep(3)

            # 检查进程是否还在运行
            if process.poll() is None:
                print("✅ 服务器成功启动")

                # 停止服务器
                try:
                    process.terminate()
                    time.sleep(2)

                    if process.poll() is None:
                        process.kill()

                    process.wait(timeout=5)
                    print("✅ 服务器已停止")
                except Exception as e:
                    print(f"❌ 停止服务器时出错: {e}")
            else:
                print(f"❌ 服务器启动失败，退出码: {process.returncode}")

        except Exception as e:
            print(f"❌ 启动服务器时出错: {e}")

    # 最终检查端口状态
    print(f"\n🔍 最终端口状态检查...")
    final_occupied = check_port(host, port)
    if final_occupied:
        print("⚠️  端口仍被占用")
    else:
        print("✅ 端口已释放")

    print("\n🎉 端口清理测试完成")

if __name__ == "__main__":
    test_port_cleanup()