	. /etc/profile.modules

# remove all previously loaded modules
# (this avoids conflicts)
	module purge

# add gcc and nvcc
	module load gcc/4.6.2
	module load cuda/5.5	

# add emacs for convenience
# (not necessary)
	module load editor/emacs/23.1
