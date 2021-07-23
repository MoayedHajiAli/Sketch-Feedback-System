import pickle

class Utils:
    """
    General utils to be used
    """
    @staticmethod
    def save_obj_pkl(dictionary, path):
        with open(path, 'wb') as f:
            pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)
