if [ ! -f "_smatch.so" ]; then
	echo "compiling fast smatch"
	python2 setup.py build
	mv build/*/_smatch.so .
	rm -rf build
else
	echo "using fast smatch"
fi
PYTHONPATH=. python2 fast_smatch.py --pr -f $1 $2
