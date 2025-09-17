#!/usr/bin/env python3
"""
新的DQPSK仿真测试程序 - 使用usrp_test中的dqpsk_system.py
对比两个版本的性能差异
"""

import numpy as np
import time
import sys
import os

from dqpsk_system import USRP_DQPSK_System

if __name__ == "__main__":
    us=USRP_DQPSK_System()
    us.transmit_and_receive()