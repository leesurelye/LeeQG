# -*- coding: utf-8 -*-
"""
@Time    : 2021/9/14 20:53
@Author  : Le Yu E
@FileName: ParameterError.py
@Software: PyCharm
"""
class ParameterError(Exception):
	def __init__(self,  *args):
		super().__init__(self)
		self.error = f"{args} parameter is required!"


	def __str__(self):
		return self.error

class CheckParameterError(Exception):
	def __init__(self, parameter, choice):
		super(CheckParameterError, self).__init__()
		self.error = f'[{parameter}] parameter Error, choice:{choice}'