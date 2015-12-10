__author__ = 'victor'

"Small p experiment"
seed=14578396
np.random.seed(seed)
x1 = np.zeros(shape=(100, 40))
p1 = x1.shape[0]
p2 = x1.shape[1]
for i in range(0,p2):
	x1[:,i] = np.asarray([math.floor(a) for a in np.random.sample(p1)*100])
	x1[np.where(x1[:,i]>85),i] = np.nan

x1_imputed, seed = imputeknn(x1)

"Large p experiment"
x2 = np.zeros(shape=(2183,60))
p1 = x2.shape[0]
p2 = x2.shape[1]
for i in range(0,p2):
	x2[:,i] = np.asarray([math.floor(a) for a in np.random.sample(p1)*100])
	x2[np.where(x2[:,i]>85),i] = np.nan

x2_imputed, seed = imputeknn(x2)

