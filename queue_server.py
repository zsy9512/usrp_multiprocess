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

def main():
    parser = argparse.ArgumentParser(description="USRP DQPSK队列服务器")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=50000, help="服务器端口")
    parser.add_argument("--authkey", type=str, default="queue_key", help="认证密钥")

    args = parser.parse_args()

    # 注册队列
    QueueManager.register('get_queue', callable=get_queue)

    # 创建管理器
    manager = QueueManager(address=(args.host, args.port), authkey=args.authkey.encode())

    print("=== USRP DQPSK队列服务器 ===")
    print(f"服务器地址: {args.host}:{args.port}")
    print(f"认证密钥: {args.authkey}")
    print("启动服务器...")

    try:
        manager.start()
        print("队列服务器已启动，等待客户端连接...")
        print("按Ctrl+C停止服务器")

        # Keep running with interrupt check
        while True:
            import time
            time.sleep(0.1)  # Small sleep to allow interrupt checking

    except KeyboardInterrupt:
        print("\n收到停止信号，正在关闭服务器...")
    finally:
        manager.shutdown()
        print("服务器已关闭")

if __name__ == "__main__":
    # 设置多进程启动方法（Windows兼容）
    multiprocessing.set_start_method('spawn', force=True)
    main()