#!/usr/bin/env python3
"""
DQPSK连续传输实验快速启动脚本
一键启动完整的多进程通信实验

支持模式：
- simulation: 仿真模式，用于测试算法
- hardware: 硬件USRP模式，用于实际信号收发
- transceiver: 自收自发模式，合并发射和接收功能

更新日期: 2025-09-17
"""

import subprocess
import time
import sys
import os
import argparse
import signal
import threading
import atexit

class DQPSKExperiment:
    """DQPSK实验管理器"""

    def __init__(self, mode="simulation", host="127.0.0.1", port=50000):
        # 参数验证
        if mode not in ["simulation", "hardware", "transceiver"]:
            raise ValueError(f"无效的模式: {mode}，必须是 'simulation', 'hardware' 或 'transceiver'")
        
        if not isinstance(port, int) or port <= 0 or port > 65535:
            raise ValueError(f"无效的端口号: {port}，必须是 1-65535 之间的整数")
        
        self.mode = mode
        self.host = host
        self.port = port
        self.processes = []
        self.threads = []
        
        # 注册退出时的清理函数
        atexit.register(self._emergency_cleanup)

        # 实验参数
        self.params = {
            "tx_freq": 900e6,
            "rx_freq": 900e6,
            "rate": 1e6,
            "tx_gain": 40,
            "rx_gain": 50,
            "frame_interval": 0.1
        }

    def start_queue_server(self):
        """启动队列服务器"""
        print("🚀 启动队列服务器...")
        cmd = [
            sys.executable, "queue_server.py",
            "--host", self.host,
            "--port", str(self.port)
        ]

        process = subprocess.Popen(cmd, cwd=os.getcwd())
        self.processes.append(("Queue Server", process))

        # 等待服务器启动
        time.sleep(2)
        return process.poll() is None

    def start_processing_program(self):
        """启动处理程序（带GUI）"""
        print("📊 启动处理程序（GUI模式）...")
        cmd = [
            sys.executable, "processing_program.py",
            "--mode", self.mode,
            "--ipc_mode", "queue",
            "--host", self.host,
            "--port", str(self.port),
            "--rate", str(self.params["rate"])
        ]

        process = subprocess.Popen(cmd, cwd=os.getcwd())
        self.processes.append(("Processing Program", process))

        # 等待GUI启动
        time.sleep(3)
        return process.poll() is None

    def start_rx_program(self):
        """启动接收程序"""
        print("📡 启动接收程序...")
        cmd = [
            sys.executable, "rx_program.py",
            "--mode", self.mode,
            "--ipc_mode", "queue",
            "--host", self.host,
            "--port", str(self.port),
            "--rx_freq", str(self.params["rx_freq"]),
            "--rate", str(self.params["rate"]),
            "--rx_gain", str(self.params["rx_gain"])
        ]

        # 硬件模式添加额外参数
        if self.mode == "hardware":
            cmd.extend(["--buffer_size", "20000"])

        process = subprocess.Popen(cmd, cwd=os.getcwd())
        self.processes.append(("RX Program", process))

        time.sleep(2)
        return process.poll() is None

    def start_transceiver_program(self):
        """启动自收自发程序"""
        print("📻 启动自收自发程序...")
        cmd = [
            sys.executable, "transceiver_program.py",
            "--tx_freq", str(self.params["tx_freq"]),
            "--rx_freq", str(self.params["rx_freq"]),
            "--rate", str(self.params["rate"]),
            "--tx_gain", str(self.params["tx_gain"]),
            "--rx_gain", str(self.params["rx_gain"]),
            "--args", "name=MyB210",
            "--buffer_size", "50000",
            "--repeat_count", "10"
        ]

        process = subprocess.Popen(cmd, cwd=os.getcwd())
        self.processes.append(("Transceiver Program", process))

        time.sleep(2)
        return process.poll() is None

    def start_tx_program(self):
        """启动发射程序"""
        print("📤 启动发射程序...")
        cmd = [
            sys.executable, "tx_program.py",
            "--mode", self.mode,
            "--ipc_mode", "queue",
            "--host", self.host,
            "--port", str(self.port),
            "--tx_freq", str(self.params["tx_freq"]),
            "--rate", str(self.params["rate"]),
            "--tx_gain", str(self.params["tx_gain"])
        ]

        process = subprocess.Popen(cmd, cwd=os.getcwd())
        self.processes.append(("TX Program", process))

        time.sleep(2)
        return process.poll() is None

    def check_processes(self):
        """检查所有进程状态"""
        print("\n📋 进程状态检查:")
        all_running = True

        for name, process in self.processes:
            if process.poll() is None:
                print(f"✅ {name}: 运行中 (PID: {process.pid})")
            else:
                print(f"❌ {name}: 已停止 (退出码: {process.returncode})")
                all_running = False

        return all_running

    def start_experiment(self):
        """启动完整实验"""
        print("🎯 开始DQPSK连续传输实验")
        print("=" * 50)
        print(f"模式: {self.mode}")
        print(f"服务器: {self.host}:{self.port}")
        print(f"参数: {self.params}")
        print("=" * 50)

        # 设置信号处理器
        def signal_handler(signum, frame):
            print(f"\n🛑 收到信号 {signum}，正在关闭...")
            self.stop_experiment()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            # 根据模式选择启动步骤
            if self.mode == "transceiver":
                steps = [
                    ("队列服务器", self.start_queue_server),
                    ("处理程序", self.start_processing_program),
                    ("自收自发程序", self.start_transceiver_program)
                ]
            else:
                steps = [
                    ("队列服务器", self.start_queue_server),
                    ("处理程序", self.start_processing_program),
                    ("接收程序", self.start_rx_program),
                    ("发射程序", self.start_tx_program)
                ]

            for step_name, step_func in steps:
                print(f"\n🔄 {step_name}...")
                if not step_func():
                    print(f"❌ {step_name}启动失败")
                    return False
                print(f"✅ {step_name}启动成功")

            # 等待系统稳定
            print("\n⏳ 等待系统稳定...")
            time.sleep(5)

            # 最终状态检查
            if self.check_processes():
                print("\n🎉 实验启动成功！")
                print("📺 GUI界面应该已经打开")
                print("📊 观察实时数据处理和星座图")
                print("⚡ 按Ctrl+C停止实验")

                # 保持运行并监控
                self.monitor_experiment()
            else:
                print("\n❌ 部分组件启动失败，请检查日志")

        except KeyboardInterrupt:
            print("\n🛑 收到停止信号，正在关闭...")
            self.stop_experiment()
        except Exception as e:
            print(f"\n❌ 实验启动失败: {e}")
            self.stop_experiment()

    def monitor_experiment(self):
        """监控实验运行状态"""
        import threading
        
        # 创建停止事件
        stop_event = threading.Event()
        
        # 设置信号处理器
        def signal_handler(signum, frame):
            print(f"\n🛑 收到信号 {signum}，正在停止监控...")
            stop_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            print("\n🔍 开始监控实验状态...")
            print("📊 每10秒检查一次进程状态")
            print("⚡ 按Ctrl+C停止实验")
            
            while not stop_event.is_set():
                # 检查进程状态
                if not self.check_processes():
                    print("⚠️  检测到进程异常，实验可能已停止")
                    break
                
                # 等待10秒或直到收到停止信号
                stop_event.wait(10)
                
        except Exception as e:
            print(f"\n❌ 监控过程中发生错误: {e}")
        finally:
            print("\n🔚 停止实验监控")
            self.stop_experiment()

    def stop_experiment(self):
        """停止所有进程"""
        print("\n🔄 正在停止实验...")

        # 第一阶段：优雅终止 (SIGTERM)
        print("📤 发送终止信号到所有进程...")
        for name, process in self.processes:
            try:
                if process.poll() is None:
                    print(f"🛑 发送SIGTERM到 {name} (PID: {process.pid})...")
                    if hasattr(process, 'terminate'):
                        process.terminate()
            except Exception as e:
                print(f"❌ 发送SIGTERM到 {name} 时出错: {e}")

        # 等待进程响应SIGTERM
        print("⏳ 等待进程优雅退出...")
        time.sleep(3)

        # 第二阶段：检查哪些进程仍在运行
        still_running = []
        for name, process in self.processes:
            try:
                if process.poll() is None:
                    still_running.append((name, process))
                    print(f"⚠️  {name} 仍在运行，准备强制终止")
                else:
                    print(f"✅ {name} 已优雅退出")
            except Exception as e:
                print(f"❌ 检查 {name} 状态时出错: {e}")

        # 第三阶段：强制终止仍在运行的进程 (SIGKILL)
        if still_running:
            print("💀 强制终止仍在运行的进程...")
            for name, process in still_running:
                try:
                    print(f"� 发送SIGKILL到 {name} (PID: {process.pid})...")
                    if hasattr(process, 'kill'):
                        process.kill()
                except Exception as e:
                    print(f"❌ 发送SIGKILL到 {name} 时出错: {e}")

            # 等待强制终止完成
            print("⏳ 等待强制终止完成...")
            time.sleep(2)

            # 检查强制终止结果
            for name, process in still_running:
                try:
                    if process.poll() is None:
                        print(f"❌ {name} 仍然在运行 (PID: {process.pid})")
                    else:
                        print(f"✅ {name} 已强制停止")
                except Exception as e:
                    print(f"❌ 检查强制终止结果时出错: {e}")

        # 最终状态报告
        final_running = []
        for name, process in self.processes:
            try:
                if process.poll() is None:
                    final_running.append(f"{name}(PID:{process.pid})")
            except:
                pass
        
        if final_running:
            print(f"⚠️  以下进程可能仍在后台运行: {', '.join(final_running)}")
            print("💡 提示: 可以使用任务管理器手动终止这些进程")
            print("🔍 进程PID列表: " + ", ".join([f"{name}({pid})" for name, pid in [(n, p.pid) for n, p in self.processes if p.poll() is None]]))
        else:
            print("✅ 所有进程已成功停止")

        print("🏁 实验已完全停止")

    def _emergency_cleanup(self):
        """紧急清理函数，在程序异常退出时调用"""
        print("\n🚨 检测到程序异常退出，正在紧急清理...")
        
        # 记录所有进程的PID用于调试
        process_info = []
        for name, process in self.processes:
            try:
                if process.poll() is None:
                    process_info.append(f"{name}(PID:{process.pid})")
                    print(f"💀 紧急终止 {name} (PID: {process.pid})")
                    process.kill()
                else:
                    print(f"✅ {name} 已退出 (PID: {process.pid})")
            except Exception as e:
                print(f"❌ 紧急清理 {name} 时出错: {e}")
        
        if process_info:
            print(f"🔍 已尝试终止的进程: {', '.join(process_info)}")
        else:
            print("✅ 所有进程已正常退出")
        
        print("🧹 紧急清理完成")

def main():
    parser = argparse.ArgumentParser(description="DQPSK连续传输实验启动器")
    parser.add_argument("--mode", choices=["simulation", "hardware", "transceiver"],
                       default="simulation", help="运行模式 (transceiver为自收自发模式)")
    parser.add_argument("--host", default="127.0.0.1", help="服务器主机")
    parser.add_argument("--port", type=int, default=50000, help="服务器端口")
    parser.add_argument("--tx-freq", type=float, default=900e6, help="发射频率 (Hz)")
    parser.add_argument("--rx-freq", type=float, default=900e6, help="接收频率 (Hz)")
    parser.add_argument("--rate", type=float, default=1e6, help="采样率 (Hz)")
    parser.add_argument("--tx-gain", type=float, default=40, help="发射增益 (dB)")
    parser.add_argument("--rx-gain", type=float, default=50, help="接收增益 (dB)")

    args = parser.parse_args()

    # 创建实验实例
    experiment = DQPSKExperiment(
        mode=args.mode,
        host=args.host,
        port=args.port
    )

    # 更新参数
    experiment.params.update({
        "tx_freq": args.tx_freq,
        "rx_freq": args.rx_freq,
        "rate": args.rate,
        "tx_gain": args.tx_gain,
        "rx_gain": args.rx_gain
    })

    # 启动实验
    experiment.start_experiment()

if __name__ == "__main__":
    main()