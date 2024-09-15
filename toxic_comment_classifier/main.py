from src.predictor import Predictor

def main():
    predictor = Predictor()
    while True:
        print('------------------------------------------------')
        print('Enter the comments: ', end='')
        text = input()
        output = predictor.predict(text)
        print('Model Output: ', output)

if __name__ == "__main__":
    main()