# Get the maximum vertical velocity this run
# $1 = .leave file

ed -s $1 <<\EOF
?Max w this run?
+2
q
EOF
