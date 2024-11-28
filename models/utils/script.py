
def func(path):

    with open(path, 'r') as table:
        with open('table1', 'a') as _table:
            line = table.readline()
            cnt = 0
            while line:
                _table.write(f'\'{line.strip()}\', ')
                cnt += 1
                if not (cnt % 10):
                    _table.write('\n')
                line = table.readline()
                
if __name__ == '__main__':
    path = './tokens_table'
    func(path)