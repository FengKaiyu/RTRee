

f = open('Singapore.poi.data', 'r')
fo = open('unprocessed.data', 'w')
fo2 = open('processed.data', 'w')
fo3 = open('poi.data', 'w')
is_in = False
content = []
lat = None
lon = None
category = None
for line in f:
	line = line.strip()
	if line.startswith('<node id='):
		eles = line.split(' ')
		lat = eles[2][5:-1]
		lon = eles[3][5:-1]
		if is_in:
			content.clear()
			content.append(line)
		else:
			is_in = True
			content.append(line)
	elif line.startswith('</node>'):
		if not is_in:
			print('error')
			print(content)
			print(line)
		else:
			is_in = False
			if category is None:
				for v in content:
					fo.write('{}\n'.format(v))
				fo.write('{}\n'.format(line))
			else:
				for v in content:
					fo2.write('{}\n'.format(v))
				fo2.write('{}\n'.format(line))
				fo3.write('{} {} {}\n'.format(lat, lon, str(category)))
			content.clear()
			category = None
	if line.startswith('<tag k='):
		eles = line.split(' ')
		key = eles[1][3:-1]
		value = ' '.join(eles[2:])[3:-3]
		if key == 'name' or key == 'amenity' or key == 'building' or key == 'building:use' or key == 'shop' or key == 'tourism':
			if category is None:
				category = {}
			category[key] = value
		content.append(line)

fo2.close()
fo3.close()
f.close()
fo.close()