# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
def format_dict(dictionary):
    """
    creates a formatted string that represents a dictionary on multiple lines

    >>> a = "{'d': 4, 'b': 2, 'c': 3, 'a': {'d': 4, 'b': 2, 'c': 3, 'a': {'d': 4, 'b': 2, 'c': 3, 'a': 1}}}"
    >>> a
    {
    'a' : {
          'a' : {
                'a' : 1,
                'b' : 2,
                'c' : 3,
                'd' : 4,},
          'b' : 2,
          'c' : 3,
          'd' : 4,},
    'b' : 2,
    'c' : 3,
    'd' : 4,}
    """
    formatted = ""
    for key in sorted(dictionary):
        val = dictionary[key]

        if isinstance(key,str):
            key = "'{}'".format(key)

        if isinstance(val,dict):
            val = format_dict(val).replace('\n','\n' + ' ' * (len(key)+3))

        formatted += "\n{} : {},".format(key,val)

    return '{' + formatted + '}'

def main():
    import copy
    import imsciutils as iu
    a = {'a':1,'b':2,'c':3,'d':4}
    b = copy.deepcopy(a)
    b['a'] = copy.deepcopy(a)
    b['a']['a'] = copy.deepcopy(a)
    print( iu.util.format_dict(b) )

if __name__ == "__main__":
    main()
