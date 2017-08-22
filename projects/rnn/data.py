import codecs

class Data:
    def __init__(self, filename):
        with codecs.open(filename, 'r', 'utf-8') as f:
            self.data = f.read()
        chars = list(set(self.data))
        self.size, self.vocab_size = len(self.data), len(chars)
        self.char_to_ix = { ch:i for i,ch in enumerate(chars) }
        self.ix_to_char = { i:ch for i,ch in enumerate(chars) }
        print 'data has %d characters, %d unique.' % (self.size, self.vocab_size)

    def part(self, start, end):
        return [self.char_to_ix[ch] for ch in self.data[start:end]]

    def txt(self, indices):
        return ''.join(self.ix_to_char[ix] for ix in indices)
