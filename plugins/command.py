"""
Command 插件，用于处理命令行输入和执行相应的操作。
"""
from typing import Optional
import inspect
import json
import os
import logging


class Command:
    def __init__(self, controller, commands_file='plugins/commands.json'):
        self.controller = controller
        self.commands_file = commands_file
        self.commands = {}
        self.descriptions = {}
        self.no_param_commands = set()
        self.logger = logging.getLogger(__name__)
        self._load_commands()

    def _load_commands(self):
        """从外部文件加载命令和描述"""
        if not os.path.exists(self.commands_file):
            raise FileNotFoundError(f"Commands file '{self.commands_file}' not found.")
        try:
            with open(self.commands_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            commands_data = data.get('commands', {})
            descriptions_data = data.get('descriptions', {})
            no_param_commands = data.get('no_param_commands', [])
            for cmd, method_str in commands_data.items():
                try:
                    method = getattr(self.controller, method_str.split('.')[-1])
                    self.commands[cmd] = method
                    self.descriptions[cmd] = descriptions_data.get(cmd, '无描述')
                    self.logger.info(f"Loaded command '{cmd}' -> {method_str}")
                except AttributeError as e:
                    self.logger.error(f"Failed to load command '{cmd}': method '{method_str}' not found on controller. Skipping.")
            self.no_param_commands = set(no_param_commands)
            self.logger.info(f"Loaded {len(self.commands)} commands, {len(self.no_param_commands)} no-param commands.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing commands from '{self.commands_file}': {e}")

    def _convert_arg(self, arg_str: str):
        """尝试将字符串参数转换为适当的类型"""
        try:
            # 尝试转换为 int
            if arg_str.isdigit() or (arg_str.startswith('-') and arg_str[1:].isdigit()):
                return int(arg_str)
            # 尝试转换为 float
            float_val = float(arg_str)
            return float_val
        except ValueError:
            pass
        # 默认返回字符串
        return arg_str

    def execute(self, command_str: str) -> str:
        """执行命令字符串，支持参数"""
        parts = command_str.strip().split()
        if not parts:
            return "错误: 命令为空"

        cmd = parts[0].lower()
        if cmd not in self.commands:
            return f"错误: 未知命令 '{cmd}'。可用命令: {', '.join(self.commands.keys())}"

        args = parts[1:]  # 获取参数
        try:
            func = self.commands[cmd]
            sig = inspect.signature(func)
            param_count = len(sig.parameters)

            if len(args) > 0:
                # 检查函数是否接受参数
                if cmd in self.no_param_commands:
                    return f"错误: 命令 '{cmd}' 不支持参数"
                if param_count == 0:
                    return f"错误: 命令 '{cmd}' 不支持参数"
                if len(args) > param_count:
                    return f"错误: 命令 '{cmd}' 期望 {param_count} 个参数，但提供了 {len(args)} 个"
                if len(args) < param_count:
                    return f"错误: 命令 '{cmd}' 期望 {param_count} 个参数，但提供了 {len(args)} 个"
                # 转换参数类型
                converted_args = [self._convert_arg(arg) for arg in args]
                result = func(*converted_args)
                self.logger.info(f"Executed command '{cmd}' with args {converted_args}")
            else:
                if param_count > 0 and cmd not in self.no_param_commands:
                    return f"错误: 命令 '{cmd}' 需要参数"
                result = func()
                self.logger.info(f"Executed command '{cmd}' without args")

            # 处理返回值
            if result is not None:
                return f"命令 '{cmd}' 执行成功，结果: {result}"
            else:
                return f"命令 '{cmd}' 执行成功"
        except Exception as e:
            self.logger.error(f"Error executing command '{cmd}': {e}")
            return f"错误: 执行命令 '{cmd}' 时发生异常: {type(e).__name__}: {e}"

    def list_commands(self) -> str:
        """列出所有可用命令及其描述"""
        lines = ["可用命令:"]
        for cmd in self.commands.keys():
            desc = self.descriptions.get(cmd, "无描述")
            param_info = ""
            if cmd in self.no_param_commands:
                param_info = " (无参数)"
            else:
                sig = inspect.signature(self.commands[cmd])
                param_count = len(sig.parameters)
                if param_count > 0:
                    param_info = f" (需要 {param_count} 个参数)"
                else:
                    param_info = " (无参数)"
            lines.append(f"- {cmd}{param_info}: {desc}")
        return "\n".join(lines)

    def register_command(self, cmd, method, description, no_param=False):
        """动态注册新命令"""
        self.commands[cmd] = method
        self.descriptions[cmd] = description
        if no_param:
            self.no_param_commands.add(cmd)
        self.logger.info(f"Registered new command '{cmd}' (no_param: {no_param})")
