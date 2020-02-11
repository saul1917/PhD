import numpy as np

class ClassHierarchy():

    """
    Args:
        class_level_names (tuple or list): Index
    """
    def __init__(self, class_level_names):
        self.class_level_names = class_level_names
        self.class_level_size = len(class_level_names)
        self.class_names = {}
        for i in range(self.class_level_size):
            self.class_names[class_level_names[i]] = []
        self.hierarchy_matrix = None

    def _hierarchy_is_empty(self):
        return self.hierarchy_matrix is None

    """
    Args:
        entry (1 dimension nparray): Hierarchy entry
    Returns:
        True or False
    """
    def _entry_exists(self, entry):
        return np.any(np.equal(self.hierarchy_matrix, entry).all(1))

    #get the idx of a class level name
    def _get_idx_of_class_level(self, class_level_name):
        if class_level_name in self.class_level_names:
            return self.class_level_names.index(class_level_name)

    #get an id of a class given the class level name
    def _get_idx_by_class_level_name(self, class_level_name, class_name):
        class_level_idx = self._get_idx_of_class_level(class_level_name)
        return self._get_idx_by_class_level_idx(class_level_idx, class_name)

    #get an id of a class given the class level idx, not the name
    def _get_idx_by_class_level_idx(self, class_level_idx, class_name):
        class_names = self.class_names[self.class_level_names[class_level_idx]]
        if class_name in class_names:
            return class_names.index(class_name)
        else:
            return -1

    #adds anew class to the dictionary if it doesnt exist, given a class level name
    def add_class_by_name(self, class_level_name, class_name):
        idx = self._get_idx_by_class_level_name(class_level_name, class_name)
        if idx==-1:
            self.class_names[class_level_name] += [class_name]
            return self.class_names[class_level_name].index(class_name)
        return idx

    #entry must be a np array
    def add_hierarchy_entry(self, entry):
        #entry must have the number of class levels defined during construction
        assert(len(entry) == self.class_level_size), "Hierarchy entry size must match the number of class levels"
        assert(isinstance(entry, np.ndarray)), "Hierarchy entry must be a numpy array"
        if self._hierarchy_is_empty():
            self.hierarchy_matrix = np.array([entry])
        else:
            if not self._entry_exists(entry):
                self.hierarchy_matrix = np.vstack((self.hierarchy_matrix, entry))

    def get_children_idx_at_class_level_name(self, parent_level_name, parent_class_name, child_level_name):
        parent_level_idx = self._get_idx_of_class_level(parent_level_name)
        parent_idx = self._get_idx_by_class_level_idx(parent_level_idx, parent_class_name)
        assert(parent_idx > -1), "Parent level name must exist in the class level names."
        child_level_idx = self._get_idx_of_class_level(child_level_name)
        return self.get_children_idx_at_class_level_idx(parent_level_idx, parent_idx, child_level_idx)

    def get_children_names_at_class_level_name(self, parent_level_name, parent_class_name, child_level_name):
        idxs = self.get_children_idx_at_class_level_name(parent_level_name, parent_class_name, child_level_name)
        return np.array(self.class_names[child_level_name])[idxs]

    def get_children_idx_at_class_level_idx(self, parent_level_idx, parent_idx, children_level_idx):
        return np.unique(np.array(self.hierarchy_matrix[self.hierarchy_matrix[:,parent_level_idx]==parent_idx][:,children_level_idx]))

    def get_class_level_size(self, idx):
        return len(self.class_names[self.class_level_names[idx]])

    def get_hierarchy_mask(self, parent_level, children_level):
        parent_size = len(self.class_names[self.class_level_names[parent_level]])
        children_size = len(self.class_names[self.class_level_names[children_level]])
        mask = np.zeros((parent_size, children_size))
        children_idxs = self.hierarchy_matrix[:, children_level]
        parent_idxs = self.hierarchy_matrix[:, parent_level]
        mask[parent_idxs,children_idxs] = 1.0
        return mask

    def __repr__(self):
        fmt_str = 'Hierarchy of Classes ' + self.__class__.__name__ + '\n'
        fmt_str += 'Number of Class Levels: {}\n'.format(self.class_level_size)
        fmt_str += 'Class levels:\n'
        for i in self.class_level_names:
            fmt_str += '    {}:{}\n'.format(i, len(self.class_names[i]))
        fmt_str += str(self.hierarchy_matrix)
        return fmt_str
