# from h5py._hl.base import KeysViewHDF5
from h5py import File


def show_keys(data, level=0):
    # print('  ' * level + key)
    if not hasattr(data, 'keys'):  #  and isinstance(data.keys(), KeysViewHDF5)):
        if hasattr(data, 'shape'):
            print('  ' * level + str(data.shape))
        return

    for key in data.keys():
        print('  ' * level + key)

        show_keys(data[key], level=level + 1)


# mofa_data = File('models/pyro-MOFA.h5')
# show_keys(mofa_data)