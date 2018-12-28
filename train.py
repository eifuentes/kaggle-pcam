data_dir = '/Users/eifuentes/Data/pcam/'
batch_size = 8
num_workers = 8
validation_size = 0.10
seed = None

label_filename = Path(data_dir, 'train_labels.csv')
label_df = pd.read_csv(label_filename)
num_samples = label_df['label'].shape[0]
label_cnts = label_df['label'].value_counts()
print(f'Negative Class ({100*label_cnts[0]/num_samples:.3f}%): {label_cnts[0]}')
print(f'Postive Class ({100*label_cnts[1]/num_samples:.3f}%): {label_cnts[1]}')
print(f'Number of Samples: {num_samples}')

training_dset = PCamDataset(data_dir, 'train', label_filename, transform=vtransforms.ToTensor())
num_val_samples = int(np.floor(len(training_dset) * validation_size))
num_train_samples = len(training_dset) - num_val_samples
train_dset, val_dset = random_split(training_dset, [num_train_samples, num_val_samples])
test_dset = PCamDataset(data_dir, 'test', transform=vtransforms.ToTensor())
train_dloader = DataLoader(train_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
val_dloader = DataLoader(val_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
test_dloader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
train_mean, train_std = calculate_statistics(train_dloader)
val_mean, val_std = calculate_statistics(val_dloader)
test_mean, test_std = calculate_statistics(test_dloader)


train_dtransform = vtransforms.Compose([
    vtransforms.ColorJitter(brightness=0.05, contrast=0.05),
    vtransforms.RandomAffine(degrees=(180), translate=(0.05, 0.05), scale=(0.8, 1.2), shear=0.05, resample=PIL.Image.BILINEAR),
    vtransforms.RandomHorizontalFlip(),
    vtransforms.RandomVerticalFlip(),
    vtransforms.ToTensor(),
    vtransforms.Normalize(train_mean, train_std)
])
test_dtransform = vtransforms.Compose([
    vtransforms.ToTensor(),
    vtransforms.Normalize(train_mean, train_std)
])

training_dset = PCamDataset(data_dir, 'train', label_filename, transform=train_dtransform)
num_val_samples = int(np.floor(len(training_dset) * validation_size))
num_train_samples = len(training_dset) - num_val_samples
train_dset, val_dset = random_split(training_dset, [num_train_samples, num_val_samples])
test_dset = PCamDataset(data_dir, 'test', transform=test_dtransform)

train_dloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
val_dloader = DataLoader(val_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
test_dloader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

x, y = next(iter(train_dloader))
vtransforms.ToPILImage()(vutils.make_grid(x, nrow=8, normalize=True, scale_each=False))

seed = seed if seed else randint(1, 1000)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda')
    non_blocking = True
else:
    device = torch.device('cpu')
    non_blocking = False