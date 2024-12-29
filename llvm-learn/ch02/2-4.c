int factor(int n) {
    int ret = 1;
    while (n > 1) {
        ret *= n;
        n--;
    }
    return ret;
}


// opt -passes=dot-cfg 2-4.ll -f
