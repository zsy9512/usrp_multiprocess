#!/usr/bin/env python3
"""
Queue Server: Provides a shared queue for multiprocess communication.
Clients: rx_program.py and processing_program.py
"""

import multiprocessing
from multiprocessing.managers import BaseManager
import argparse
import sys
import os
import signal
import threading
import socket
import time
import subprocess

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class QueueManager(BaseManager):
    """队列管理器"""
    pass

# 创建全局共享队列
shared_queue = multiprocessing.Queue(maxsize=1000)

def get_queue():
    """返回共享队列"""
    return shared_queue

def check_port_available(host, port):
    """检查端口是否可用（区分LISTENING和TIME_WAIT状态）"""
    try:
        # 首先尝试socket连接测试
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            # 端口被占用，检查是否是LISTENING状态
            try:
                netstat_result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
                if netstat_result.returncode == 0:
                    lines = netstat_result.stdout.split('\n')
                    for line in lines:
                        if f'{host}:{port}' in line and 'LISTENING' in line:
                            return False  # 有进程在监听，端口不可用
                    return True  # 只有TIME_WAIT等状态，可以重用
                else:
                    return False  # 无法确定状态，保守起见认为不可用
            except:
                return False  # 无法检查状态，保守起见认为不可用
        else:
            return True  # 端口完全可用
    except:
        return True  # 连接测试失败，认为端口可用

def kill_process_on_port(port):
    """尝试终止占用指定端口的进程（Windows）"""
    try:
        import subprocess
        # 使用 netstat 查找占用端口的进程
        result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
        killed_pids = []
        
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        if pid not in killed_pids:  # 避免重复终止同一进程
                            try:
                                # 终止进程
                                subprocess.run(['taskkill', '/PID', pid, '/F'], capture_output=True)
                                killed_pids.append(pid)
                                print(f"✅ 已终止占用端口 {port} 的进程 (PID: {pid})")
                            except Exception as e:
                                print(f"❌ 终止进程 {pid} 失败: {e}")
        
        if killed_pids:
            print(f"🔍 已终止 {len(killed_pids)} 个进程: {', '.join(killed_pids)}")
            time.sleep(3)  # 等待进程完全终止
            return True
        else:
            print(f"ℹ️  未找到占用端口 {port} 的进程")
            return False
            
    except Exception as e:
        print(f"❌ 检查端口占用失败: {e}")
        return False

def signal_handler(signum, frame):
    """信号处理器"""
    print(f"\n🛑 收到信号 {signum}，正在关闭服务器...")
    print("📊 正在清理队列和连接...")
    raise KeyboardInterrupt

def main():
    # 设置信号处理器
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description="USRP DQPSK队列服务器")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=50000, help="服务器端口")
    parser.add_argument("--authkey", type=str, default="queue_key", help="认证密钥")
    parser.add_argument("--force", action="store_true", help="强制清理占用端口的进程")

    args = parser.parse_args()

    # 检查端口是否可用
    print(f"🔍 检查端口 {args.host}:{args.port} 是否可用...")
    if not check_port_available(args.host, args.port):
        print(f"⚠️  端口 {args.port} 已被占用")
        
        if args.force:
            print("🔧 尝试清理占用端口的进程...")
            if kill_process_on_port(args.port):
                print("✅ 端口清理完成，等待2秒...")
                time.sleep(2)
            else:
                print("❌ 无法清理端口占用，请手动终止相关进程")
                return
        else:
            print("💡 使用 --force 参数强制清理，或手动终止占用端口的进程")
            print("🔍 查找占用端口的进程命令: netstat -ano | findstr :50000")
            return
    
    # 注册队列
    QueueManager.register('get_queue', callable=get_queue)

    # 创建管理器
    manager = None
    try:
        manager = QueueManager(address=(args.host, args.port), authkey=args.authkey.encode())

        print("=== USRP DQPSK队列服务器 ===")
        print(f"服务器地址: {args.host}:{args.port}")
        print(f"认证密钥: {args.authkey}")
        print("启动服务器...")

        manager.start()
        print("✅ 队列服务器已启动，等待客户端连接...")
        print("⚡ 按Ctrl+C停止服务器")
        print("🔍 监控队列状态...")

        # 使用事件来控制循环
        stop_event = threading.Event()
        
        def signal_handler_thread(signum, frame):
            """线程中的信号处理器"""
            print(f"\n🛑 线程收到信号 {signum}")
            stop_event.set()
        
        # 在主线程中设置信号处理器
        signal.signal(signal.SIGTERM, signal_handler_thread)
        signal.signal(signal.SIGINT, signal_handler_thread)

        # 主循环 - 更频繁地检查停止事件
        while not stop_event.is_set():
            time.sleep(0.05)  # 更短的睡眠时间，更快响应信号
            
            # 定期报告队列状态
            try:
                queue_size = shared_queue.qsize()
                if queue_size > 0:
                    print(f"📊 队列大小: {queue_size}")
            except:
                pass  # 队列可能被其他进程修改

    except KeyboardInterrupt:
        print("\n🛑 收到停止信号，正在关闭服务器...")
        try:
            stop_event.set()
        except NameError:
            pass  # stop_event可能未定义
    except Exception as e:
        print(f"❌ 服务器错误: {e}")
        try:
            stop_event.set()
        except NameError:
            pass  # stop_event可能未定义
    finally:
        # 安全地关闭管理器
        if manager is not None:
            try:
                print("🔄 正在关闭管理器...")
                if hasattr(manager, 'shutdown'):
                    manager.shutdown()
                    print("✅ 管理器已关闭")
                else:
                    print("⚠️  管理器没有shutdown方法，尝试其他清理方式")
                    
                # 清理队列
                try:
                    while not shared_queue.empty():
                        shared_queue.get_nowait()
                    print("🧹 队列已清理")
                except:
                    pass
                    
            except Exception as e:
                print(f"❌ 关闭管理器时出错: {e}")
        else:
            print("ℹ️  管理器未创建，无需清理")
                
        print("🏁 服务器已完全关闭")

if __name__ == "__main__":
    # 设置多进程启动方法（Windows兼容）
    multiprocessing.set_start_method('spawn', force=True)
    main()