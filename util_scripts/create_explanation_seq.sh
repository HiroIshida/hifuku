n_grid=600
echo -e "step1\nstep2\nstep3" | parallel python3 visualize_2dtask_result.py --save -n 0 -mode {} -grid $n_grid
for ((i=1; i<=7; i++))
do
    echo -e "step0\nstep1\nstep2\nstep3" | parallel python3 visualize_2dtask_result.py --save -n $i -mode {} -grid $n_grid
done

for image_file in figs/*.{jpg,jpeg,png,gif}; do
    [ -f "$image_file" ] || continue  # Skip if it's not a regular file
    convert "$image_file" -rotate 90 "${image_file%.*}-rotated.png"
    rm $image_file
    echo "Image '$image_file' rotated 90 degrees clockwise."
done
