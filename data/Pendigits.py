from torch.utils.data import DataLoader, Dataset
import numpy as np


class Pendigits(Dataset):
    def __init__(self, file_path: str):
        lines = []
        classes = []
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                coordinates = list(map(int, line.rstrip().split(',')[:-1]))
                _coordinates = []
                for j in range(0, len(coordinates), 2):
                    _coordinates.append(coordinates[j: j + 2])
                coordinates = _coordinates
                line_class = int(line.split(',')[-1])
                lines.append(coordinates)
                classes.append(line_class)

        self.classes = np.array(classes)
        self.lines = np.array(lines)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        return self.lines[item], self.classes[item]


if __name__ == '__main__':
    ds = Pendigits('pendigits.tra')
    dl = DataLoader(ds, num_workers=2, batch_size=10)
    for idx, e in enumerate(dl):
        l, c = e
        print(c.shape)
        print(l.shape)
        break
