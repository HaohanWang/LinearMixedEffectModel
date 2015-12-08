__author__ = 'haohanwang'

import numpy as np


class PLinkFormatReader():
    def __init__(self, path=None):
        self.path = path

    def read_gene_tpedFile(self, geneFile, header):
        m = 1 if header else 0
        text = [line.strip() for line in open(self.path + geneFile)][m:]
        n = len(text)*2
        m = (len(text[0].split()) - 4)/2
        data = np.zeros([m, n])
        for i in range(len(text)):
            items = text[i].split()
            for j in range((len(items)-4)/2):
                data[j, i*2] = float(items[2*j + 4])
                data[j, i*2+1] = float(items[2*j+1 + 4])
        data = np.matrix(data)
        # print data.shape
        return data



    def read_phenoFile(self, phenoFile, col, header):
        m = 1 if header else 0
        text = [line.strip() for line in open(self.path + phenoFile)][m:]
        data = []
        for line in text:
            items = line.split()
            data.append(float(items[col]))
        return np.array(data)

    def readFile(self, gene1File, gene2File, phenoFile, phenoCol=2, geneFileHeader=False, phenoFileHeader=True):
        gene1 = self.read_gene_tpedFile(gene1File, geneFileHeader)
        gene2 = self.read_gene_tpedFile(gene2File, geneFileHeader)
        pheno = self.read_phenoFile(phenoFile, phenoCol, phenoFileHeader)
        return gene1, gene2, pheno

