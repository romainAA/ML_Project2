import numpy as np

def in_fun(test,central,margin):
    """ Helper function to check if test is in the interval [central-margin;central+margin]"""
    #if test < central + np.sqrt(margin)*100 and test > central - np.sqrt(margin)*100:
    #if test < central + margin and test > central - margin:
    if test < central + np.sqrt(margin) and test > central - np.sqrt(margin):
        return True
    return False

def add_neighbors(X, patch_size):
    """ Add a feature to patches that compare the mean of grey of a patch with his neighbors

    We consider each patch and check if any of his neighbors has the same mean of grey
    more or less his standard deviation. 
    """
    Xbis = np.zeros((X.shape[0],3))
    tmp = (X.shape[0]%100)/patch_size
    for i in range(len(X)):
        if i%patch_size == 0 :
            if i%len(X)/100 < tmp:
                if in_fun(X[i+1][0],X[i][0],X[i][1]) or in_fun(X[int(i+tmp)][0],X[i][0],X[i][1]):
                        Xbis[i] = np.append(X[i],1)
            elif i%len(X)/100 > (patch_size-1)*tmp:
                if in_fun(X[i+1][0],X[i][0],X[i][1]) or in_fun(X[int(i-tmp)][0],X[i][0],X[i][1]):
                        Xbis[i] = np.append(X[i],1)
            else:
                if in_fun(X[i+1][0],X[i][0],X[i][1]) or in_fun(X[int(i-tmp)][0],X[i][0],X[i][1]) or in_fun(X[int(i+tmp)][0],X[i][0],X[i][1]):
                        Xbis[i] = np.append(X[i],1)
        elif (i+1)%patch_size == 0:
            if i%len(X)/100 < tmp:
                if in_fun(X[i-1][0],X[i][0],X[i][1]) or in_fun(X[int(i+tmp)][0],X[i][0],X[i][1]):
                        Xbis[i] = np.append(X[i],1)
            elif i%len(X)/100 > (patch_size-1)*tmp:
                if in_fun(X[i-1][0],X[i][0],X[i][1]) or in_fun(X[int(i-tmp)][0],X[i][0],X[i][1]):
                        Xbis[i] = np.append(X[i],1)
            else:
                if in_fun(X[i-1][0],X[i][0],X[i][1]) or in_fun(X[int(i-tmp)][0],X[i][0],X[i][1]) or in_fun(X[int(i+tmp)][0],X[i][0],X[i][1]):
                        Xbis[i] = np.append(X[i],1)
        else:
            if i%len(X)/100 < tmp:
                if in_fun(X[i-1][0],X[i][0],X[i][1]) or in_fun(X[i+1][0],X[i][0],X[i][1]) or in_fun(X[int(i+tmp)][0],X[i][0],X[i][1]):
                        Xbis[i] = np.append(X[i],1)
            elif i%len(X)/100 > (patch_size-1)*tmp:
                if in_fun(X[i-1][0],X[i][0],X[i][1]) or in_fun(X[i+1][0],X[i][0],X[i][1]) or in_fun(X[int(i-tmp)][0],X[i][0],X[i][1]):
                        Xbis[i] = np.append(X[i],1)
            elif in_fun(X[i-1][0],X[i][0],X[i][1]) or in_fun(X[i+1][0],X[i][0],X[i][1]) or in_fun(X[int(i-tmp)][0],X[i][0],X[i][1]) or in_fun(X[int(i+tmp)][0],X[i][0],X[i][1]):
                        Xbis[i] = np.append(X[i],1)

    return Xbis
