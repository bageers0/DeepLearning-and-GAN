import os
import model
def main():
    if __name__ == '__main__':
        paths="./output"#your output path
        if not os.path.exists(path=paths):
            os.makedirs(paths)
        with tf.Graph().as_default():
            gan=model.Gannetworks(10000,0.001,64)
            gan.model_train()
main()
