import numpy as np
from imageio.v3 import imread
import matplotlib.pyplot as plt

#######################################################################
# Part 1 - numpy
def get_highest_weight_loss_participant(training_data, participant_names):
    return participant_names[np.argmax(training_data[:, 0] - training_data[:, -1])]  



def get_diff_data(training_data):
    return training_data[:, 1:] - training_data[:, :-1]


def get_highest_change_month(training_data, study_months):
    diff_matrix = np.abs(training_data[:, 1:] - training_data[:, :-1])
    total_changes = np.sum(diff_matrix, axis=0)
    return study_months[np.argmax(total_changes)]

def get_inconsistent_participants(training_data, participant_names):
    inconsistent = np.any(training_data[:, 1:] >= training_data[:, :-1], axis=1)
    return [participant_names[i] for i in range(len(participant_names)) if inconsistent[i]]

#######################################################################
# Part 2 - numpy image processing

def read_image(img_path, mode='L'):
    img = imread(img_path, mode=mode)
    return (img.astype(np.uint8))

def naive_blending(img1, img2, mask):
    return np.where(mask == 255, img1, img2)

def blur_and_downsample(img, nb_size=5):
    img_blurred = blur_image(img, nb_size)
    img_downsampled = img_blurred[::2, ::2]
    return img_downsampled.astype(np.uint8)

def get_neighborhood(img, x, y, nb_size):
    return img[max(0, x - nb_size):min(img.shape[0], x + nb_size + 1), max(0, y - nb_size):min(img.shape[1], y + nb_size + 1)]

def blur_image(im, nb_size):
    new_im=np.zeros(im.shape)
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            neighborhood = get_neighborhood(im, x, y, nb_size)
            new_im[x,y] = np.mean(neighborhood)
    return new_im.astype(np.uint8)


def build_gaussian_pyramid(img, max_levels, nb_size=5):
    pyramid = [img]
    for i in range(1, max_levels):
        newimg = blur_and_downsample(pyramid[-1], nb_size)
        pyramid.append(newimg)
    return pyramid


def upsample_and_blur(img, nb_size=5):
    duplicated_rows = np.repeat(img, 2, axis=0)
    duplicated_rows_and_columns = np.repeat(duplicated_rows, 2, axis=1)
    blurred_img = blur_image(duplicated_rows_and_columns, nb_size)
    return blurred_img.astype(np.uint8)

def build_laplacian_pyramid(img, max_levels, nb_size=5):
    g_pyramid = build_gaussian_pyramid(img, max_levels, nb_size)
    l_pyramid = []
    
    for i in range(len(g_pyramid) - 1):
        upsampled = upsample_and_blur(g_pyramid[i + 1], nb_size)
        laplacian = g_pyramid[i] - upsampled[:g_pyramid[i].shape[0], :g_pyramid[i].shape[1]]
        l_pyramid.append(laplacian)
    
    l_pyramid.append(g_pyramid[-1]) 
    return l_pyramid


def laplacian_pyramid_to_image(laplacian_pyramid, nb_size=5):
    img = laplacian_pyramid[-1]  
    for i in range(len(laplacian_pyramid) - 2, -1, -1): 
        upsampled = upsample_and_blur(img, nb_size)
        img = upsampled[:laplacian_pyramid[i].shape[0], :laplacian_pyramid[i].shape[1]] + laplacian_pyramid[i]
    return img.astype(np.uint8)

def pyramid_blending(img1, img2, mask, max_levels, nb_size=5):
    laplacian_pyramid1 = build_laplacian_pyramid(img1, max_levels, nb_size)
    laplacian_pyramid2 = build_laplacian_pyramid(img2, max_levels, nb_size)
    
    binary_mask = np.where(mask != 0, 255, 0).astype(np.uint8)
    
    gaussian_pyramid_mask = build_gaussian_pyramid(binary_mask, max_levels, nb_size)
    
    blended_pyramid = []
    for i in range(max_levels):
        mask_normalized = (gaussian_pyramid_mask[i].astype(np.float64) / 255 > 0).astype(np.float64)
        
        blended_level = (mask_normalized * laplacian_pyramid1[i].astype(np.float64) +
                         (1 - mask_normalized) * laplacian_pyramid2[i].astype(np.float64))
        blended_pyramid.append(blended_level.astype(np.uint8))
    
    blended_image = laplacian_pyramid_to_image(blended_pyramid, nb_size)
    
    return blended_image.astype(np.uint8)



############ Bonus ############

def pyramid_blending_RGB_image(img1, img2, mask, max_levels, nb_size=5):
    blended_r = pyramid_blending(img1[:, :, 0], img2[:, :, 0], mask, max_levels, nb_size)
    blended_g = pyramid_blending(img1[:, :, 1], img2[:, :, 1], mask, max_levels, nb_size)
    blended_b = pyramid_blending(img1[:, :, 2], img2[:, :, 2], mask, max_levels, nb_size)
    
    blended_img = np.stack((blended_r, blended_g, blended_b), axis=2)
    return blended_img



if __name__ == '__main__':
    def array_compare(a, b, threshold=1e-10):
        if a.shape != b.shape:
            return False
        return np.abs(a - b).mean() < threshold

    #######################################################################
    # Q1 checks
    print("Starting Q1 checks")
    training_data = np.loadtxt('training_data.csv', delimiter=',')
    participant_names = ['Jane', 'Naomi', 'John', 'Moshe']
    study_months = ['November', 'December', 'January', 'February', 'March', 'April']

    diff_data_expected = np.array([[-2.7, 1.5, -2.7, -2.7, -2.2],
                                   [-4.4, -0.2, -0.7, -1.5, -1.4],
                                   [-1.0, -1.2, 0.6, -0.3, -1.6],
                                   [-2.5, -4.1, -3.1, -2.7, -2.8]])

    print(get_highest_weight_loss_participant(training_data, participant_names) == 'Moshe')
    print(array_compare(get_diff_data(training_data), diff_data_expected))
    print(get_highest_change_month(training_data, study_months) == 'November')
    print(set(get_inconsistent_participants(training_data, participant_names)) == set(['Jane', 'John']))

    training_data = np.array([[92.3, 91.5, 89.4, 89.2, 88.6, 85.9],
                              [104.6, 102.1, 100.8, 98.5, 96.3, 94.4],
                              [73.2, 72.6, 72.0, 71.4, 71.2, 70.9],
                              [78.3, 78.0, 77.2, 75.8, 74.7, 74.4]])

    diff_data_expected = np.array([[-0.8, -2.1, -0.2, -0.6, -2.7],
                                   [-2.5, -1.3, -2.3, -2.2, -1.9],
                                   [-0.6, -0.6, -0.6, -0.2, -0.3],
                                   [-0.3, -0.8, -1.4, -1.1, -0.3]])

    print(get_highest_weight_loss_participant(training_data, participant_names) == 'Naomi')
    print(array_compare(get_diff_data(training_data), diff_data_expected))
    print(get_highest_change_month(training_data, study_months) == 'March')
    print(get_inconsistent_participants(training_data, participant_names) == [])

    #######################################################################
    # Q2 checks
    print("Starting Q2 checks")
    plot_flag = True  # Convert to False to not show plots

    def compare_images(img, img_path, mode):
        if mode == 'RGB':
            plt.imsave(f"stud_{img_path}", img)
        else:
            plt.imsave(f"stud_{img_path}", img, cmap='gray')

        stud_naive_blended_im = imread(f"stud_{img_path}", mode=mode) / 255
        gt_naive_blended_im = imread(f"results_for_presubmit_tests/{img_path}", mode=mode) / 255
        print(array_compare(gt_naive_blended_im, stud_naive_blended_im, threshold=1/254))


    # Q1
    print("Q1.1")
    img1 = read_image('apple.png')
    img2 = read_image('orange.png')
    mask = read_image('mask.png')

    print(img1.shape == img2.shape == mask.shape == (448, 624))

    max_levels = 5
    nb_size = 5

    if plot_flag:
        import matplotlib.patches as patches

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(img1, cmap='gray')
        axes[0].axis('off')
        axes[0].title.set_text('Original Image 1')
        axes[1].imshow(img2, cmap='gray')
        axes[1].axis('off')
        axes[1].title.set_text('Original Image 2')
        axes[2].imshow(mask, cmap='gray')  # cmap is required for presenting a grayscale image
        axes[2].axis('off')
        axes[2].title.set_text('Mask')
        border = patches.Rectangle((0, 0), mask.shape[1], mask.shape[0],
                                   linewidth=3, edgecolor='black', facecolor='none')
        axes[2].add_patch(border)
        plt.show()

    # Q2
    print("Q1.2")
    naive_blended_img = naive_blending(img1, img2, mask)
    compare_images(naive_blended_img, "orapple_naive.png", mode='L')

    if plot_flag:
        plt.imshow(naive_blended_img, cmap='gray')
        plt.title("Q2 - naive blending")
        plt.axis('off')
        plt.show()

    # Q3.a
    print("Q1.3.a")
    blurred_img = blur_and_downsample(img1, 5)
    print(blurred_img.shape == (224, 312))
    compare_images(blurred_img, "blurred_apple.png", mode='L')
    if plot_flag:
        plt.imshow(blurred_img, cmap='gray')
        plt.title("Q3.a - image blur")
        plt.axis('off')
        plt.show()

    # Q3.b
    print("Q1.3.b")
    im1_gp = build_gaussian_pyramid(img1, 3, 5)
    print(len(im1_gp) == 3)

    for i in range(3):
        print(im1_gp[i].shape == tuple(s // (2 ** i) for s in img1.shape))
        compare_images(im1_gp[i], f"gp3_im{i}.png", mode='L')

    # Q3.c
    print("Q1.3.c")
    upsampled_im = upsample_and_blur(img1, 5)
    print(upsampled_im.shape == tuple(s * 2 for s in img1.shape))
    compare_images(upsampled_im, "upsampled_im.png", mode='L')

    # Q3.d
    print("Q1.3.d")
    im1_lp = build_laplacian_pyramid(img1, 3, 5)
    print(len(im1_lp) == 3)

    for i in range(3):
        print(im1_lp[i].shape == tuple(s // (2 ** i) for s in img1.shape))
        compare_images(im1_lp[i], f"lp3_im{i}.png", mode='L')

    # Q3.e
    print("Q1.3.e")
    reconstructed_img = laplacian_pyramid_to_image(im1_lp, 5)
    print(reconstructed_img.shape == img1.shape)
    compare_images(reconstructed_img, "reconstructed_apple.png", mode='L')

    # Q3.f
    print("Q1.3.f")
    blended_im_gray = pyramid_blending(img1, img2, mask, max_levels, nb_size)
    compare_images(blended_im_gray, "blended_im_gray.png", mode='L')
    if plot_flag:
        plt.imshow(blended_im_gray, cmap='gray')
        plt.title("Q3.f - image blending")
        plt.axis('off')
        plt.show()

    # Q3.g (Bonus)
    print("======= Bonus Part =======")
    print("Q1.3.g")
    img1 = read_image('apple.png', 'RGB')
    img2 = read_image('orange.png', 'RGB')
    mask = read_image('mask.png', 'L')
    print(img1.shape == img2.shape == (448, 624, 3))
    print(mask.shape == (448, 624))
    blended_im = pyramid_blending_RGB_image(img1, img2, mask, max_levels, nb_size)
    compare_images(blended_im, "orapple.png", mode='RGB')

    if plot_flag:
        plt.imshow(blended_im)
        plt.title("Q3.g - RGB image blending")
        plt.axis('off')
        plt.show()
