import graphlab as gl


def main():
    rcmder = gl.load_model('model')
    rcmder.recommend(['glennq', 'neonh'], k=30).print_rows(60, 4)


if __name__ == '__main__':
    main()
