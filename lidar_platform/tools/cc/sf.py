import os


from ...config.config import cc_custom, cc_std, cc_exe
from .. import misc
from .CCCommand import CCCommand


def sf_add_const(filename, name_value, in_place=False,
                 silent=True, verbose=False, global_shift='AUTO', fmt='sbf'):
    """
    Add one constant or several constant scalar fields to a point cloud

    :param filename: the path of the point cloud
    :param name_value: either a (str, float/int) or a list of (str, float/int)
        (("spam", 0.5), ("eggs", 5))
        ("spam", 0.5)
    :param in_place: do not create a copy of the cloud, try to save the result in place
    :param silent: do not display CloudCompare console
    :param verbose: see messages in the python console (set silent to True)
    :param global_shift: CloudCompare global_shift
    :param fmt: export format (see CloudCompare documentation)
    :return: the name of the output cloud
    """

    root, ext = os.path.splitext(filename)
    if in_place:
        out = root + '.' + fmt.lower()
    else:
        out = root + '_SF_ADD_CONST.' + fmt.lower()

    cmd = CCCommand(cc_exe, silent=silent, fmt=fmt)
    cmd.open_file(filename, global_shift=global_shift)

    if isinstance(name_value, (tuple, list)):
        if isinstance(name_value[0], (tuple, list)) and len(name_value[0]) == 2:
            for name, value in name_value:
                if type(name) is not str:
                    raise TypeError('Name must be a string')
                if type(value) is not float and type(value) is not int:
                    raise TypeError('Value must be a number')
                cmd.extend(['-SF_ADD_CONST', name, str(value)])
        elif type(name_value[0]) is str:
            if isinstance(name_value[1], (float, int)):
                cmd.extend(['-SF_ADD_CONST', name_value[0], str(name_value[1])])
            else:
                raise TypeError('Value must be a number')
        else:
            raise ValueError('name_value shall be either a (str, float/int) or a list of (str, float/int)')
    else:
        raise ValueError('name_value shall be either a (str, float/int) or a list of (str, float/int)')

    cmd.extend(['-SAVE_CLOUDS', 'FILE', out])

    misc.run(cmd, verbose=verbose)

    return out
