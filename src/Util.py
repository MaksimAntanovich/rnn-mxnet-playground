from Const import args_bptt

class Util:
    @staticmethod
    def batchify(data, batch_size):
        """Reshape data into (num_example, batch_size)"""
        nbatch = data.shape[0] // batch_size
        data = data[:nbatch * batch_size]
        data = data.reshape((batch_size, nbatch)).T
        return data

    @staticmethod
    def get_batch(source, i):
        seq_len = min(args_bptt, source.shape[0] - 1 - i)
        data = source[i: i + seq_len]
        target = source[i + 1: i + 1 + seq_len]
        return data, target.reshape((-1,))

    @staticmethod
    def detach(hidden):
        if isinstance(hidden, (tuple, list)):
            hidden = [i.detach() for i in hidden]
        else:
            hidden = hidden.detach()
        return hidden
