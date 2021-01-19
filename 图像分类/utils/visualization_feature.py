def mkdir(path):

    isExists = os.path.exists(path) # 判断路径是否存在，若存在则返回True，若不存在则返回False
    if not isExists: # 如果不存在则创建目录
        os.makedirs(path)
        return True
    else:
        return False


    # 可视化每一层的feature
    for i, feature in enumerate(features):
        feature = feature[0].cpu().detach().numpy()
        print(i)
        for channel in range(feature.shape[0]):
            print(channel)
            feature_channel = feature[channel].squeeze()
            feature_channel = (feature_channel - np.amin(feature_channel))/(np.amax(feature_channel) - np.amin(feature_channel) + 1e-5) # 注意要防止分母为0！ 
            feature_channel = np.round(feature_channel * 255) # [0, 1]——[0, 255],为cv2.imwrite()函数而进行

            mkdir('checkpoints/1/features/' + str(i))  # 创建保存文件夹，以选定可视化层的序号命名
            cv2.imwrite('checkpoints/1/features/' + str(i) + '/' + str(channel) + '.jpg', feature_channel)  
