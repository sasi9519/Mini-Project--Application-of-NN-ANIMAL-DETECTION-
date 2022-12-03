# Mini-Project--Application-of-NN

## Project Title:
 ANIMAL DETECTION USING CNN
## Project Description
Animal Detection solution is a State-of-the-art Computer Vision and Artificial Intelligence based application that enables you to monitor your animal population with ease using traditional mounted cameras or drones to capture video feeds. Our Animal Detection solution will help keep track of your animals and provide an overall count.
## Algorithm:
    Step 1:
    Read the csv file and create the Data frame using pandas.
    Step 2:
    Select the " Open " column for prediction. Or select any column of your interest
    and scale the values using MinMaxScaler.
    Step 3:
    Create two lists for X_train and y_train. And append the collection of 60 readings
    in X_train, for which the 61st reading will be the first output in y_train.
    Step 4:
    Create a model with the desired number of nuerons and one output neuron.
    Step 5:
    Follow the same steps to create the Test data. But make sure you combine the
    training data with the test data.
    Step 6:
    Make Predictions and plot the graph with the Actual and Predicted values.

## Program:

    import numpy as np
    import pickle
    import cv2
    from os import listdir
    from sklearn.preprocessing import LabelBinarizer
    from keras.models import Sequential
    from keras.layers.normalization import BatchNormalization
    from keras.layers.convolutional import Conv2D
    from keras.layers.convolutional import MaxPooling2D
    from keras.layers.core import Activation, Flatten, Dropout, Dense
    from keras import backend as K
    from keras.preprocessing.image import ImageDataGenerator
    from keras.optimizers import Adam
    from keras.preprocessing import image
    from keras.preprocessing.image import img_to_array
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    EPOCHS = 25
    INIT_LR = 1e-3
    BS = 32
    default_image_size = tuple((256, 256))
    image_size = 0
    directory_root = '../input/plantvillage/'
    width=256
    height=256
    depth=3
    def convert_image_to_array(image_dir):
        try:
            image = cv2.imread(image_dir)
            if image is not None :
                image = cv2.resize(image, default_image_size)   
                return img_to_array(image)
            else :
                return np.array([])
        except Exception as e:
            print(f"Error : {e}")
            return None
    image_list, label_list = [], []
    try:
        print("[INFO] Loading images ...")
        root_dir = listdir(directory_root)
        for directory in root_dir :
            # remove .DS_Store from list
            if directory == ".DS_Store" :
                root_dir.remove(directory)

        for plant_folder in root_dir :
            plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")

            for disease_folder in plant_disease_folder_list :
                # remove .DS_Store from list
                if disease_folder == ".DS_Store" :
                    plant_disease_folder_list.remove(disease_folder)

            for plant_disease_folder in plant_disease_folder_list:
                print(f"[INFO] Processing {plant_disease_folder} ...")
                plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")

                for single_plant_disease_image in plant_disease_image_list :
                    if single_plant_disease_image == ".DS_Store" :
                        plant_disease_image_list.remove(single_plant_disease_image)

                for image in plant_disease_image_list[:200]:
                    image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"
                    if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                        image_list.append(convert_image_to_array(image_directory))
                        label_list.append(plant_disease_folder)
        print("[INFO] Image loading completed")  
    except Exception as e:
        print(f"Error : {e}")
    image_size = len(image_list)
    label_binarizer = LabelBinarizer()
    image_labels = label_binarizer.fit_transform(label_list)
    pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
    n_classes = len(label_binarizer.classes_)
    print(label_binarizer.classes_)
    np_image_list = np.array(image_list, dtype=np.float16) / 225.0
    print("[INFO] Spliting data to train, test")
    x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42) 
    aug = ImageDataGenerator(
        rotation_range=25, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, 
        zoom_range=0.2,horizontal_flip=True, 
        fill_mode="nearest")
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1
    model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes))
    model.add(Activation("softmax"))
    model.summary()
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    # distribution
    model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
    # train the network
    print("[INFO] training network...")
    history = model.fit_generator(
        aug.flow(x_train, y_train, batch_size=BS),
        validation_data=(x_test, y_test),
        steps_per_epoch=len(x_train) // BS,
        epochs=EPOCHS, verbose=1
        )
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    #Train and validation accuracy
    plt.plot(epochs, acc, 'b', label='Training accurarcy')
    plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
    plt.title('Training and Validation accurarcy')
    plt.legend()

    plt.figure()
    #Train and validation loss
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.show()
    print("[INFO] Calculating model accuracy")
    scores = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {scores[1]*100}")
    print("[INFO] Saving model...")
    pickle.dump(model,open('cnn_model.pkl', 'wb'))
    loaded_model = pickle.load(open('cnn_model.pkl', 'rb'))
    loaded_model = pickle.load(open('path\\cnn_model.pkl', 'rb'))
    image_dir="path\\plantdisease_dataset\\PlantVillage\\Potato___Early_blight"

    im=convert_image_to_array(image_dir)
    np_image_li = np.array(im, dtype=np.float16) / 225.0
    npp_image = np.expand_dims(np_image_li, axis=0)
    result=model.predict(npp_image)

    print(result)
    itemindex = np.where(result==np.max(result))
    print("probability:"+str(np.max(result))+"\n"+label_binarizer.classes_[itemindex[1][0]])



## Output:
   ![Screenshot (34)](https://user-images.githubusercontent.com/83326978/205449157-a17a0dfa-ba00-4f9f-8ed3-2fc58a235311.png)


![Screenshot (35)](https://user-images.githubusercontent.com/83326978/205449168-7603d445-e5a6-44d9-b014-9c6719d94a45.png)


![Screenshot (36)](https://user-images.githubusercontent.com/83326978/205449172-189f2be5-8f2c-4d35-9f16-e22977707e76.png)

    ELEPHANT:
  ![a](https://user-images.githubusercontent.com/83326978/205448426-7ec21d59-14ce-438d-976c-c30cb9e1db08.png)
  


## Advantage :
This algorithm classifies animals based on their images so we can monitor them more efficiently. Animal detection and classification can help to prevent animal-vehicle accidents, trace animals and prevent theft. This can be achieved by applying effective deep learning algorithms.
## Result:
Thus,the program has been implemented successfully and the output was obtained.
