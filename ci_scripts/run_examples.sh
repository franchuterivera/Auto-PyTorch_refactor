cd examples

for script in *.py
do
    echo '###############################################################################'
    echo '###############################################################################'
    echo "Starting to test $script"
    echo '###############################################################################'
    python $script
    rval=$?
    if [ "$rval" != 0 ]; then
        echo "Error running example $script"
        exit $rval
    fi
done
