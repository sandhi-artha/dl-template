from config import CFG
from model.model import MyCNN


def run():
    """Builds model, loads data, trains and evaluates"""
    print('initializing..')
    model = MyCNN(CFG)
    print('loading data..')
    model.load_data()
    # model.check_data()
    print('creating model..')
    model.build()
    # print('starting training..')
    # model.train()
    print('eval results:')
    model.evaluate()


if __name__ == '__main__':
    run()