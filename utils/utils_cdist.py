class SimpleConverterForCDistNet(object):
    def __init__(self, alphabet):
        self.alphabet = alphabet + '-'
        self.dict = {i: char for i, char in enumerate(self.alphabet)}

    def decode(self, t):
        """Decode tensor to string."""
        texts = []
        for i in range(t.size(0)):
            text = []
            for j in range(t.size(1)):
                char_idx = t[i, j].item()
                if char_idx < len(self.alphabet) - 1:
                    text.append(self.dict[char_idx])
                else:
                    break 
            texts.append(''.join(text))
        return texts