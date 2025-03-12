import os

class CCCommand(list):

    def __init__(self, cc_exe, silent=True, auto_save='OFF', fmt='SBF'):
        self.append(cc_exe)
        if silent:
            self.append('-SILENT')
        self.append('-NO_TIMESTAMP')
        if auto_save.lower() == 'off':
            self.extend(['-AUTO_SAVE', 'OFF'])
        self.append('-C_EXPORT_FMT')
        if fmt.lower() == 'laz':  # needed to export to laz /!\ OLD SYNTAX, not with new las/laz plugin qLASIO /!\
            self.append('LAS')
            self.append("-EXT")
            self.append("laz")
        else:
            self.append(fmt)

    def open_file(self, fullname, global_shift='AUTO', fwf=False):
        if not os.path.exists(fullname):
            raise FileNotFoundError(fullname)
        if fwf:
            self.append('-fwf_o')  # old syntax for full waveform, only for backward compatibility
        else:
            self.append('-o')
        if global_shift is not None:
            self.append('-GLOBAL_SHIFT')
            if global_shift == 'AUTO' or global_shift == 'FIRST':
                self.append(global_shift)
            elif type(global_shift) is tuple or type(global_shift) is list:
                x, y, z = global_shift
                self.append(str(x))
                self.append(str(y))
                self.append(str(z))
            else:
                raise ValueError('invalid value for global_shit')
        self.append(fullname)