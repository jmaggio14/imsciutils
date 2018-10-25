# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
from __future__ import print_function
from termcolor import colored

def get_printer(name, log_level='info'):
    """
    creates or retrieves a printer object

    Args:
        name (str): name of the Printer to create or retrieve
        log_level (str,int): local log level for the Printer

    Returns:
        iu.Printer: printer with the given name
    """
    if name in Printer.ACTIVE_PRINTERS:
        return Printer.ACTIVE_PRINTERS[name]
    else:
        return Printer(name, log_level)

def disable_all_printers():
    """disables all printers
    sets the global log level to infinity to effectively disable all
    printers
    """
    set_global_printout_level( float('inf') )


def whitelist_printer(name):
    """
    whitelisted the given printer

    only whitelisted printers will print to the terminal

    Args:
        name (str): name of the printer to whilelist
    """
    # remove from blacklist if it's currently on it
    if name in Printer.BLACKLIST:
        Printer.BLACKLIST.remove(name)

    # append to Whitelist if it's not already on there
    if name not in Printer.WHITELIST:
        if Printer.WHITELIST == 'all':
            Printer.WHITELIST = []
        Printer.WHITELIST.append(name)


def blacklist_printer(name):
    """
    Blacklists the given printer

    When a printer is blacklisted, it will not print out to the terminal

    Args:
        name (str): name of the printer to blacklist
    """
    # remove from whitelist if it currently exists
    if name in Printer.WHITELIST:
        Printer.WHITELIST.remove(name)

    # add to blacklist if it's not already in there
    if name not in Printer.BLACKLIST:
        Printer.BLACKLIST.append(name)

def reset_printer_lists():
    """resets the white and black lists to their default values"""
    Printer.WHITELIST = 'all'
    Printer.BLACKLIST = []

def disable_printout_colors():
    """disables colored printouts for imsciutils printout messages"""
    Printer.ENABLE_COLOR = False


def enable_printout_colors():
    """enables colored printouts for imsciutils printout messages"""
    Printer.ENABLE_COLOR = True


def get_active_printers():
    """gets a list of the names of currently active printers"""
    return sorted(Printer.ACTIVE_PRINTERS.keys())

def set_global_printout_level(log_level):
    """
    sets the global log level
    Args:
        log_level (int,str):
            the log level to set the global log level

    Returns:
        None
    """
    if log_level in Printer.LOG_LEVELS:
        log_level = Printer.LOG_LEVELS[log_level]

    if not isinstance(log_level,(int,float)):
        iu.error("unable to set log_level to {}, must be an integer".format(
                                                                    log_level))
        return

    Printer.GLOBAL_LOG_LEVEL = log_level


class Printer(object):
    """
    Object used to print out messages to the terminal for a given
    process or object. This is meant to increase traceability particularly
    for people who are developing imsciutils

    Each message printed out using this object is automatically colorized,
    filtered for the current log level, and prefixed

    """
    ACTIVE_PRINTERS = {}
    LOG_LEVELS = {
        'debug': 10,
        'info': 20,
        'warning': 30,
        'error': 40,
        'critical': 50,
        'comment': 60,
    }
    GLOBAL_LOG_LEVEL = LOG_LEVELS['info']
    WHITELIST = 'all'
    BLACKLIST = []
    ENABLE_COLOR = True
    MIN_NAME_SIZE = 18

    def __init__(self, name, log_level=0):
        if log_level in self.LOG_LEVELS:
            log_level = self.LOG_LEVELS[log_level]

        self.__error_check(name, log_level)

        self.name = name
        self.set_log_level(log_level)

        buf = ' ' * max(((self.MIN_NAME_SIZE - len(self.name)) // 2),0)
        self.level_text = {
            'debug':   '({1}{0}{1})[    DEBUG   ]'.format(self.name, buf),
            'info':    '({1}{0}{1})[    INFO    ]'.format(self.name, buf),
            'warning': '({1}{0}{1})[   WARNING  ]'.format(self.name, buf),
            'error':   '({1}{0}{1})[    ERROR   ]'.format(self.name, buf),
            'critical':'({1}{0}{1})[  CRITICAL  ]'.format(self.name, buf),
            'comment': '({1}{0}{1})[   COMMENT  ]'.format(self.name, buf),
        }

        self.ACTIVE_PRINTERS[self.name] = self

    def __error_check(self, name, log_level):
        """
        Error checks the Printer Instantiation Args
        """
        if not isinstance(log_level, (int, float)):
            raise TypeError(
                "'log_level' must be an integer or float, currently {}".format(type(log_level)))

        if not isinstance(name, str):
            raise TypeError("'name' of Printer must be a string")

    def should_print(self, message_level):
        """
        Determines whether or not the given message level should print
        to the terminal

        Args:
            message_level (str,int): level of the message

        Returns:
            bool: Whether or not the printer should print at that level
        """
        if message_level in self.LOG_LEVELS:
            message_level = self.LOG_LEVELS[message_level]

        # if message level is less than the global or local log_level
        if message_level < max(self.log_level, self.GLOBAL_LOG_LEVEL):
            return False

        # if printer is in the whitelist
        if self.WHITELIST != 'all':
            if self.name in self.WHITELIST:
                return True
            else:
                return False

        # if printer is in blacklist
        if self.BLACKLIST != []:
            if self.name not in self.BLACKLIST:
                return True
            else:
                return False

        return True

    def set_log_level(self,log_level):
        """sets the local log_level for this printer"""
        if log_level in self.LOG_LEVELS:
            log_level = self.LOG_LEVELS[log_level]
        self.__error_check(self.name, log_level)
        self.log_level = log_level


    def debug(self, *messages):
        """
        prints out objects at debug level for this printer

        Args:
            *messages: objects or messages to print out at debug level
        Returns:
            None
        """
        if not self.should_print('debug'):
            return

        prefix = self.level_text['debug']
        if self.ENABLE_COLOR:
            prefix = colored(prefix,'cyan',attrs=['bold'])
            messages = [colored(msg,'cyan',attrs=['bold']) for msg in messages]

        print(prefix, *messages)

    def info(self, *messages):
        """
        prints out objects at info level for this printer

        Args:
            *messages: objects or messages to print out at info level
        Returns:
            None
        """
        if not self.should_print('info'):
            return

        prefix = self.level_text['info']
        print(prefix, *messages)

    def warning(self, *messages):
        """
        prints out objects at warning level for this printer

        Args:
            *messages: objects or messages to print out at warning level
        Returns:
            None
        """
        if not self.should_print('warning'):
            return

        prefix = self.level_text['warning']
        if self.ENABLE_COLOR:
            prefix = colored(prefix,'yellow',attrs=['bold'])
            messages = [colored(msg,'yellow',attrs=['bold']) for msg in messages]

        print(prefix, *messages)

    def error(self, *messages):
        """
        prints out objects at error level for this printer

        Args:
            *messages: objects or messages to print out at error level
        Returns:
            None
        """
        if not self.should_print('error'):
            return

        prefix = self.level_text['error']
        if self.ENABLE_COLOR:
            prefix = colored(prefix,'red',attrs=['bold'])
            messages = [colored(msg,'red',attrs=['bold']) for msg in messages]

        print(prefix, *messages)

    def critical(self, *messages):
        """
        prints out objects at critical level for this printer

        Args:
            *messages: objects or messages to print out at critical level

        Returns:
            None
        """
        if not self.should_print('critical'):
            return

        prefix = self.level_text['critical']
        if self.ENABLE_COLOR:
            prefix = colored(prefix,'red')
            messages = [colored(msg,'red') for msg in messages]

        print(prefix, *messages)

    def comment(self, *messages):
        """
        prints out objects at comment level for this printer

        Args:
            *messages: objects or messages to print out at comment level

        Returns:
            None
        """
        if not self.should_print('comment'):
            return

        prefix = self.level_text['comment']
        if self.ENABLE_COLOR:
            prefix = colored(prefix,'green',attrs=['bold'])
            messages = [colored(msg,'green',attrs=['bold']) for msg in messages]

        print(prefix, *messages)

    def __del__(self):
        if self.name in self.ACTIVE_PRINTERS:
            del self.ACTIVE_PRINTERS[self.name]

        if self.name in self.BLACKLIST:
            self.BLACKLIST.remove(self.name)

        if self.WHITELIST != 'all':
            if self.name in self.WHITELIST:
                self.WHITELIST.remove(self.name)
