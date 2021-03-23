#TODO: Make this functional with the right imports and such
'''

    folder: The folder containing the images. Note that the other functions usually take the folder that
            contains the folder which has the images

    featureFile: The CSV file which lists which features are present in which images

    imageSize: The size of the image that we accept. A 2D tuple

    enc: The loaded encoder



def create_celeba_feature_averages(folder, featureFile, imageSize, enc):
    attributeData = pd.read_csv(featureFile)
    attributeNames = list(attributeData.columns)
    attributeNames.remove('image_id')

    featureCount = {}
    featureVectors = {}

    for attribute in attributeNames:
        featureCount[attribute] = 0
        featureVectors[attribute] = np.zeros(shape=(1, 50))

    ctr = 0

    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):

            ctr += 1

            if (ctr % 1000 == 0):
                print(f"{ctr} images processed")

            img = np.array(Image.open(
                f"{folder}/{filename}").resize(imageSize))
            img = (img / 255)
            # img = img.reshape(1, 128, 128, 3)

            _, _, z = enc(img)
            z = z.numpy()

            fileNumber = int(filename[:-4])
            featuresForImage = attributeData.iloc[fileNumber - 1]

            for attribute in attributeNames:
                if featuresForImage[attribute] == 1:
                    featureVectors[attribute] += z
                    featureCount[attribute] += 1

    for attribute in attributeNames:
        featureVectors[attribute] /= featureCount[attribute]

    return featureVectors, featureCount
'''
