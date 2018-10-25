"""
This is a script that will verify that all .py contained in our repo
contain our license and copyright.

if desired, this script can also modify each file and add the license and copyright
to each file

travis will run it with default arguments automatically, but for human use

Example:
    default (no modification, directory is ./imsciutils):
        $ python enforcer.py

    explicit path:
        $ python enforcer.py --directory='path_to_directory'

    add licenses if they don't exist:
        $ python enforcer.py --modify

"""
#!/usr/bin/env python

# MIT License
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import sys
import os
import glob
import six
import fnmatch


SKIP = ['.git',
        'LICENSE',
        'Pipfile',
        'Pipfile.lock',
        'TODO.md',
        'README.md',
        'requirements.txt']

ACCEPTABLE_EXTS = ['.py']

LICENSE_HEADER = """@Email: jmaggio14@gmail.com
@Website: https://www.imagepypelines.org/
@License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
@github: https://github.com/jmaggio14/imsciutils

Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
"""

EXTENSIONS_DICT = {'py': '#'}


def generate_header(unformatted_license, filename):
    """
    Generates the list of strings specific to the given character as
    a comment
    :returns license header lines in list
    """
    extension = filename.split('/')[-1].split('\\')[-1].split('.')[-1]
    if extension not in EXTENSIONS_DICT:
        print('Warning: unknown extension associated with:', filename)
        return []
    char = EXTENSIONS_DICT[extension]
    return [(char + ' ' + line).rstrip()
            for line in unformatted_license.split('\n')]


def check_valid_header(header, filename):
    """
    Checks to see if the given file contents has a valid header on top
    """
    with open(filename, 'r') as filestream:
        ext = os.path.splitext(filename)[1].replace('.', '')
        content = filestream.read().replace(EXTENSIONS_DICT[ext]+' ', '')
        content = content.replace(EXTENSIONS_DICT[ext], '')
        if LICENSE_HEADER not in content:
            return False
    return True


def print_valid_header(header, filename):
    """
    Simply checks to see if the header is valid, and prints accordingly
    """
    if check_valid_header(header, filename):
        print('Valid:', filename)
        return 0
    print('Invalid', filename)
    return -1


def apply_header(header, filename):
    """
    Applies the generated license header to the target file, comment
    for license header is generated through the extension dictionary
    :return True on success, false if unknown extension
    """
    if print_valid_header(header, filename) == 0:
        return 0
    with open(filename, 'r+') as filestream:
        content = filestream.read()
        filestream.seek(0)
        filestream.write("\n".join(header) + "\n")
        filestream.write(content)
    return 0


def evaluate_header(header, filename, modify):
    """
    Runs either the apply_header or print_valid_header
    depending on modify boolean
    """
    try:
        if modify:
            return apply_header(header, filename)
        return print_valid_header(header, filename)
    except UnicodeDecodeError:
        return 0


def enforce_header(unformatted_license, directory, modify=False):
    """
    Recursively iterates over target directory and finds all known
    file types and checks to see whether or not they conform to the header
    :arg modify If modify is set to True then it will inject the header if it
    does not exist
    :return 0 on success, -1 on failure
    """
    returns = 0
    if os.path.isfile(os.path.abspath(directory)):
        header = generate_header(unformatted_license, directory)
        return evaluate_header(header, directory, modify)

    matches = []
    for ext in ACCEPTABLE_EXTS:
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, '*' + ext):
                matches.append(os.path.join(root, filename))

    filenames = matches

    for obj in filenames:
        if obj not in SKIP:
            if os.path.isfile(os.path.abspath(obj)):
                header = generate_header(unformatted_license, obj)
                if evaluate_header(header, obj, modify) == -1:
                    returns = -1
            elif enforce_header(unformatted_license,
                                os.path.join(directory, obj)) == -1:
                returns = -1
    return returns


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--modify',
                        help='whether or not to modify the files in questions',
                        action='store_true',
                        )

    parser.add_argument('--directory',
                        help='the directory to enforce recursively',
                        default='./imsciutils',
                        )
    args = parser.parse_args()

    if args.modify:
        print("enforcing directory '{}' with modification".format(args.directory))
    else:
        print("enforcing directory '{}' without modification".format(args.directory))

    RESULT = enforce_header(LICENSE_HEADER, args.directory, args.modify)
    if RESULT == 0:
        print('Successful enforcement')
    else:
        print('Failed enforcement')
    sys.exit(RESULT)
