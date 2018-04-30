import pretty_midi
from torch.utils.data import Dataset


class MidiDataset(Dataset):
    """
    data  : 파일 경로 list (List of midi paths)
    example : data = ["1.mid", "2.mid" ... ]
    """
    def __init__(self, data, transform=None):
        self.samples = data
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            midi piano roll
        """
        path = self.samples[0]

        sample = pretty_midi.PrettyMIDI(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample