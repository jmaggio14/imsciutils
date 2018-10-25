# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
import os

from .Printer import get_printer

def get_default_printer():
    """gets the imsciutils default printer"""
    return get_printer('imsciutils')

def debug(*messages):
    """prints a 'debug' level message to the imsciutils default printer"""
    printer = get_default_printer()
    printer.debug(*messages)

def info(*messages):
    """prints a 'info' level message to the imsciutils default printer"""
    printer = get_default_printer()
    printer.info(*messages)

def warning(*messages):
    """prints a 'warning' level message to the imsciutils default printer"""
    printer = get_default_printer()
    printer.warning(*messages)

def error(*messages):
    """prints a 'error' level message to the imsciutils default printer"""
    printer = get_default_printer()
    printer.error(*messages)

def critical(*messages):
    """prints a 'critical' level message to the imsciutils default printer"""
    printer = get_default_printer()
    printer.critical(*messages)

def comment(*messages):
    """prints a 'comment' level message to the imsciutils default printer"""
    printer = get_default_printer()
    printer.comment(*messages)
