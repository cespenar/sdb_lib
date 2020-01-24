import numpy as np
import pandas as pd
import mesa_reader as mesa
import glob
import os
import shutil


class GyreData(object):

    """Structure containing data from a GYRE output file.

        Assumes the following structure of a file:
        
        line 1: blank
        line 2: header numbers
        line 3: header names
        line 4: header data
        line 5: body numbers
        line 6: body names
        lines 7-: body data
    
    Ispired by Bill Wolf's PyMesaReader:
    https://github.com/wmwolf/py_mesa_reader

    Parameters
    ----------
    file_name : str
        The name of GYRE output file to be read in.

    Attributes
    ----------
    file_name : str
        Path to the GYRE output file.
    bulk_data : numpy.ndarray
        The main data in the structured array format.
    bulk_names : tuple of str
        List of all available data column names.
    header_data : dict
        Header data in dict format.
    header_names : list of str
        List of all available header names.
    """

    header_names_line = 3
    body_names_line = 6

    @classmethod
    def set_header_names_line(cls, name_line=2):
        cls.header_names_line = name_line

    @classmethod
    def set_body_names_line(cls, name_line=6):
        cls.bulk_names_line = name_line
    
    def __init__(self, file_name):
        """Make a GyreData object from a GYRE output file.

        Assumes the following structure of a file:
        
        line 1: blank
        line 2: header numbers
        line 3: header names
        line 4: header data
        line 5: body numbers
        line 6: body names
        lines 7-: body data

        This structure can be altered by the class methods
        'GyreData.set_header_names_line' and 'GyreData.set_body_names_line'.

        Parameters
        ----------
        file_name : str
            The name of GYRE output file to be read in.
        """

        self.file_name = file_name
        self.header_names = None
        self.header_data = None
        self.body_names = None
        self.body_data = None
        self.read_gyre()

    def read_gyre(self):
        """Reads data from the GYRE output file."""

        self.body_data = np.genfromtxt(
            self.file_name, skip_header=GyreData.body_names_line - 1,
            names=True, dtype=None)
        
        self.body_names = self.body_data.dtype.names

        header_data = []
        with open(self.file_name) as f:
            for i, line in enumerate(f):
                if i == GyreData.header_names_line - 1:
                    self.header_names = line.split()
                elif i == GyreData.header_names_line:
                    header_data = [float(v) for v in line.split()]
                elif i > GyreData.header_names_line:
                    break
        self.header_data = dict(zip(self.header_names, header_data))




def read_progenitor_name(f_name):
    """Recovers initial values from the name of history file that
    follows specific format of progenitors' grid.

    Parameters
    ----------
    f_name : str
        Name of history file.

    Returns
    ----------
    values : dict
        Initial parameters in dict format.
	"""

    s = f_name.split('_')
    values = {}
    values['m'] = np.float(s[1][1:])
    values['rot'] = np.float(s[2][3:])
    values['z'] = np.float(s[3][1:])
    values['y'] = np.float(s[4][1:])
    values['fh'] = np.float(s[5][2:])
    values['fhe'] = np.float(s[6][3:])
    values['fenv'] = np.float(s[7][3:])
    values['mlt'] = np.float(s[8][3:])
    values['sc'] = np.float(s[9][2:])
    values['reimers'] = np.float(s[10][7:])
    values['blocker'] = np.float(s[11][7:])
    values['turbulence'] = np.float(s[12][10:])
    return values


def find_rgb_tip(data, log_Teff_max = 3.8):
    """Returns index of RGB tip.

    Parameters
    ----------
    data : MesaData
        Evolutionary track (MESA history file) as MesaData object.
    log_Teff_max : float, optional
        Maximum value of log_Teff, for which the function looks for RGB tip (max(log_L)).
        Default is 3.8.

    Returns
    ----------
    int
        Index of RGB tip.
    """

    condition = np.logical_and(data.center_h1 < 1e-4, data.log_Teff < log_Teff_max)
    return data.model_number[condition][np.argmax(data.log_L[condition])] - 1


def save_grid_to_file(grid, output):
    """Saves structured array to text file.

    Parameters
    ----------
    grid : numpy.ndarray
        Structured array with a grid of parameters.
    output : str
        The name of output file.

    Returns
    ----------
    None
    """

    format = ''
    head = ''
    for ind, name in enumerate(grid.dtype.names):
        if grid[name].dtype == np.int64:
            format = format + '%18d '
            width = 18
        elif grid[name].dtype == np.float64:
            format = format + '%18.8f '
            width = 18
        elif grid[name].dtype == 'S80':
            format = format + '%80s '
            width = 80
        if len(name) <= width:
            if ind == 0:
                head = head + (width - len(name)) * ' ' + name
            else:
                head = head + (width - len(name) + 3) * ' ' + name
        else:
            print(f"Column name {name} too long ({len(name)}) for a width: {width}")
            exit

    df = pd.DataFrame(grid)
    with open(output, 'w') as f:
        f.write(head + '\n')
    df.to_csv(output, header=False, index=False, sep='\t', float_format='%18.8f', mode='a')


def evaluate_initial_grid(grid_dir, output):
    """Reads history files in a directory tree and saves parameters to a single file.

    Parameters
    ----------
    grid_dir : str
        Path to the directory with history files.
    output : str
        The name of output file.

    Returns
    ----------
    None
    """

    history_files = glob.glob(grid_dir + '/history*')
    n = len(history_files)

    # model_number's type should be int64, but there is an issue with
    # formatting pandas output for integers
    grid = np.zeros(n, dtype= \
		[('m_i', 'float64'), \
        ('rot', 'float64'), \
        ('z', 'float64'), \
        ('y', 'float64'), \
        ('fh', 'float64'), \
        ('fhe', 'float64'), \
        ('fenv', 'float64'), \
        ('mlt', 'float64'), \
        ('sc', 'float64'), \
        ('reimers', 'float64'), \
        ('blocker', 'float64'), \
        ('turbulence', 'float64'), \
        ('m', 'float64'), \
        ('model_number', 'float64'), \
        ('log_Teff', 'float64'), \
        ('log_L', 'float64'), \
        ('age', 'float64'), \
        ('m_core', 'float64')])

    for i, track in enumerate(history_files):
        print(track)
        initial_parameters = read_progenitor_name(os.path.basename(track))
        data = mesa.MesaData(track)
        if data.center_h1[-1] < 1e-4:
            rgb_tip = find_rgb_tip(data)

            grid['m_i'][i] = initial_parameters['m']
            grid['rot'][i] = initial_parameters['rot']
            grid['z'][i] = initial_parameters['z']
            grid['y'][i] = initial_parameters['y']
            grid['fh'][i] = initial_parameters['fh']
            grid['fhe'][i] = initial_parameters['fhe']
            grid['fenv'][i] = initial_parameters['fenv']
            grid['mlt'][i] = initial_parameters['mlt']
            grid['sc'][i] = initial_parameters['sc']
            grid['reimers'][i] = initial_parameters['reimers']
            grid['blocker'][i] = initial_parameters['blocker']
            grid['turbulence'][i] = initial_parameters['turbulence']
            grid['m'][i] = data.star_mass[rgb_tip]
            grid['model_number'][i] = data.model_number[rgb_tip]
            grid['log_Teff'][i] = data.log_Teff[rgb_tip]
            grid['log_L'][i] = data.log_L[rgb_tip]
            grid['age'][i] = data.star_age[rgb_tip]
            grid['m_core'][i] = data.he_core_mass[rgb_tip]
        else:
            os.remove(track)
            print("Deleted!")
        print('')
    
    save_grid_to_file(grid, output)
    

def read_grid_file_numpy(f_name):
    """Reads a grid file and returns structured numpy arrary.

    Parameters
    ----------
    f_name : str
        A grid file.

    Returns
    ----------
    numpy.ndarray
        Numpy array with grid.
    """

    return np.genfromtxt(f_name, dtype=None, names=True)


def read_grid_file(f_name):
    """Reads a grid file. Uses pandas.

    Parameters
    ----------
    f_name : str
        A grid file.

    Returns
    ----------
    pandas.core.frame.DataFrame
        Pandas DataFrame with grid.
    """

    return pd.DataFrame(read_grid_file_numpy(f_name))


def make_replacements(file_in, file_out, replacements, remove_original=False):
    """Replaces text in an input file and creates new changed file.

    Parameters
    ----------
    file_in : str
        Input file.
    file_out : str
        Output file.
    replacements : dict
        Searched phrases and their replacements in dict format.
    remove_original : bool, optional
        Removes original if True. Default is False.

    Returns
    ----------
    None
    """

    with open(file_in) as infile, open(file_out, 'w') as outfile:
        for line in infile:
            for src, target in replacements.items():
                line = line.replace(src, target)
            outfile.write(line)
    if remove_original:
        os.remove(file_in)


def calculate_y(z, y_primordial=0.249, y_protosolar=0.2703, z_protosolar=0.0142):
    """Calculates helium abundance, Y, using given metallicity, Z.
    
    Assumed helium enrichemnt law is 1.5.

    Parameters
    ----------
    z : float
        Metallicty.
    y_primordial : float, optional
        Primordial helium abundance. Defaults to 0.249 (Planck Collaboration 2015).
    y_protosolar : float, optional
        Protosolar helium abundance. Defaults to 0.2703 (AGSS09).
    z_protosolar : float, optional
        Protosolar metallicity. Defaults to 0.0142 (AGSS09).

    Returns
    ----------
    float
        Calculated helium abundance, Y.
    """

    return y_primordial + (y_protosolar - y_primordial) * z / z_protosolar # Choi et al. (2016)


def create_workdirs_from_grid(grid, template_dir, job_description):
    """Creates MESA working directories using data from provided grid.

    Parameters
    ----------
    grid : numpy.ndarray
        Grid of models that provides parameters for created work directories.
    template_dir : str
        Template working directory.
    job_description : str
        The first part of folders' names.

    Returns
    ----------
    None
    """

    for model in grid:
        job_name = 'm' + '{:.3f}'.format(model['m_i']) + \
            '_z' + '{:.4f}'.format(model['z']) + \
            '_y' + '{:.5f}'.format(model['y']) + \
            '_rot' + '{:.1f}'.format(model['rot']) + \
            '_mlt' + '{:.2f}'.format(model['mlt']) + \
            '_sc' + '{:.3f}'.format(model['sc']) + \
            '_fh' + '{:.3f}'.format(model['fh']) + \
            '_eta' + '{:.2f}'.format(model['reimers'])
        dest_dir = job_description + '_' + job_name
        print(dest_dir)
        shutil.copytree(template_dir, dest_dir)
        shutil.move(dest_dir + '/template_run.sh', dest_dir + '/r_' + job_name + '.sh')
        replacements = { \
            '<<MASS>>':'{:.3f}'.format(model['m_i']), \
            '<<Z>>':'{:.5f}'.format(model['z']), \
            '<<Y>>':'{:.5f}'.format(model['y']), \
            '<<MLT>>':'{:.2f}'.format(model['mlt']), \
            '<<SC>>':'{:.3f}'.format(model['sc']), \
            '<<F_H>>':'{:.3f}'.format(model['fh']), \
            '<<ROT>>':'{:.1f}'.format(model['rot']), \
            '<<NUMBER>>':'{}'.format(np.int64(model['model_number'])), \
            '<<REIMERS>>':'{:.2f}'.format(model['reimers']) \
            }
        make_replacements(dest_dir + '/template_r11701.rb', dest_dir + '/job.rb', \
            replacements)


def find_semiconvection_bottom(profile):
    """Finds the zone where the natural semiconvection starts in a MESA model.

    Parameters
    ----------
    profile : mesa_reader.MesaData
        Evolutionary model (MESA profile file) as MesaData object.

    Returns
    ----------
    zone_semi_bottom : int
        The zone where the natural semiconvection starts.
    """

    zone_semi_bottom = -1
    delta_nabla = (profile.gradr - profile.gradL)[::-1]
    for i, delta in enumerate(delta_nabla):
        if i == 0:
            if delta <= 0.0:
                break
            else:
                continue
        if delta_nabla[i - 1] * delta <= 0.0:
            zone_semi_bottom = profile.zone[::-1][i - 1]
            break
    print("zone_semi_bottom =", zone_semi_bottom)
    return zone_semi_bottom


def find_semiconvection_top(profile, zone_semi_bottom):
    """Finds the zone where the natural semiconvection ends in a MESA model.

    Parameters
    ----------
    profile : mesa_reader.MesaData
        Evolutionary model (MESA profile file) as MesaData object.
    zone_semi_bottom : int
        The zone where the natural semiconvection starts.

    Returns
    ----------
    zone_semi_top : int
        The zone where the natural semiconvection ends.
    """

    ind_q08 = np.where(profile.q >= 0.8)[0][-1]
    zone_semi_top = np.argmax(profile.gradL[ind_q08:zone_semi_bottom - 1]) + ind_q08
    
    in_conv = False
    delta_nabla = (profile.gradr - profile.gradL)[zone_semi_top - 1:zone_semi_bottom - 1]
    for i, delta in enumerate(delta_nabla):
        if not in_conv and delta <= 0:
            continue
        elif not in_conv and delta > 0:
            in_conv = True
            if zone_semi_top - 1 + i >= zone_semi_bottom:
                print("zone_semi_top >= zone_semi_bottom; probably borked!")
                break
            else:
                continue
        elif in_conv and delta <= 0:
            zone_semi_top = zone_semi_top + i - 1
            break

    print("zone_semi_top =", zone_semi_top)
    return zone_semi_top


def start_semiconvection_model(history):
    """Finds the model in which the natural semiconvection emerges during the evolution.

    Parameters
    ----------
    history : mesa_reader.MesaData
        Evolutionary track (MESA history file) as MesaData object.

    Returns
    ----------
    start_model : int
        The model in which the natural semiconvection emerges.
    """

    he03 = 0.3
    he05 = 0.5
    he09 = 0.9
    ind03 = np.where(history.center_he4 < he03)[0][0]
    ind05 = np.where(history.center_he4 < he05)[0][0]
    ind09 = np.where(history.center_he4 < he09)[0][0]

    mean_mass = np.mean(history.mass_conv_core[ind05:ind03])
    std_mass = np.std(history.mass_conv_core[ind05:ind03])
    epsilon = 5.0 * std_mass
    start_model = -1
    for i, m in enumerate(history.mass_conv_core[ind09:ind03][::-1]):
        if np.abs(m - mean_mass) > epsilon:
            start_model = history.model_number[ind09:ind03][::-1][i]
            break
    print("start_model = ", start_model)

    return start_model


def end_semiconvection_model(history, eps = 5.0):
    """Finds the model in which the occurence of natural semiconvection
    during the evolution becomes ill-defined.

    Parameters
    ----------
    history : mesa_reader.MesaData
        Evolutionary track (MESA history file) as MesaData object.
    eps : float
        The parameter that controls threshold of fluctuations of
        the mass of convevective core that allows to find where
        the semiconvective zone becomes ill-defined. Guessed default
        is 5.0.

    Returns
    ----------
    end_model : int
        The model that ends the well-defined period of natural semiconvection.
    """

    he03 = 0.3
    he05 = 0.5
    ind03 = np.where(history.center_he4 < he03)[0][0]
    ind05 = np.where(history.center_he4 < he05)[0][0]

    mean_mass = np.mean(history.mass_conv_core[ind05:ind03])
    std_mass = np.std(history.mass_conv_core[ind05:ind03])
    epsilon = eps * std_mass
    end_model = -1
    for i, m in enumerate(history.mass_conv_core):
        if i < ind03:
            continue
        else:
            if ((np.abs(m - mean_mass) > epsilon) and (history.center_he4[i] > history.center_he4[i - 1])) or (m == 0.0):
                end_model = i
                break
    print("end_model = ", end_model)

    return end_model
