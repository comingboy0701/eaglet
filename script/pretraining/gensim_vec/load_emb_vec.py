# -*- coding: utf-8 -*-
import numpy as np


def load_emb_vec(word2vec_file_path: str):
    """emb path

    @param
        word2vec_file_path: emb的path
    @return:
        embedding: emb
        word2index: word转index
    """
    word2vec_file = open(word2vec_file_path, encoding='utf-8')
    word2vec_dict = dict()
    embedding_dim = -1
    for line in word2vec_file:
        line = line.strip()
        segs = line.split(" ")
        if len(segs) == 2:
            embedding_dim = int(segs[1])
            continue
        elif len(segs) != (embedding_dim + 1):
            continue
        word = segs[0]
        vector = [float(s) for s in segs[1:]]
        word2vec_dict[word] = vector
    word2vec_file.close()

    embedding = [vector for _, vector in word2vec_dict.items()]
    word2index = {word: index for index, (word, _) in enumerate(word2vec_dict.items())}

    zeros_word_vec = [0 for _ in range(0, embedding_dim)]  # 第0行填充零向量，用于尾部补齐
    avg_word_vec = list(np.mean(np.array(embedding), axis=0))  # 填充  <unk>  <blank>

    if "<PAD>" not in word2index:
        embedding.append(zeros_word_vec)
        word2index["<PAD>"] = len(embedding)-1

    if "<UNK>" not in word2index:
        embedding.append(avg_word_vec)
        word2index["<UNK>"] = len(embedding)-1

    if " " not in word2index:
        embedding.append(avg_word_vec)
        word2index[" "] = len(embedding)-1

    def swap(embedding, word2index, index2word, word, position=0):
        if word2index[word] == position:
            return embedding, word2index, index2word
        else:
            swap_word = index2word[position]
            swap_position = word2index[word]
            embedding[position], embedding[swap_position] = embedding[swap_position], embedding[position]
            word2index[word], word2index[swap_word] = word2index[swap_word],  word2index[word]
            index2word[position],  index2word[swap_position] = index2word[swap_position], index2word[position]
        return embedding, word2index, index2word

    index2word = {index: word for word, index in word2index.items()}
    embedding, word2index, index2word = swap(embedding, word2index, index2word, '<PAD>', position=0)
    embedding, word2index, index2word = swap(embedding, word2index, index2word, '<UNK>', position=1)
    embedding, word2index, index2word = swap(embedding, word2index, index2word, ' ', position=2)

    return np.array(embedding), word2index


if __name__ == '__main__':
    init_embedding, word2index = load_emb_vec("embedding/char-SougouNews.vec")

    # init_embedding, word2index_dict = load_emb_vec("embedding/char-DX1.vec")
    #
    # init_embedding, word2index_dict = load_emb_vec("embedding/char-weieryunbao.vec")
