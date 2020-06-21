#!/bin/csh
#
#  Purpose:
#
#    Create a GZIP'ed TAR file of the m_src/geometry files.
#
#  Modified:
#
#    02 January 2006
#
#  Author:
#
#    John Burkardt
#
#  Move to the directory just above the "geometry" directory.
#
cd $HOME/public_html/m_src
#
#  Delete any TAR or GZ file in the directory.
#
echo "Remove TAR and GZ files."
rm geometry/*.tar
rm geometry/*.gz
#
#  Create a TAR file of the "geometry" directory.
#
echo "Create TAR file."
tar cvf geometry_m_src.tar geometry/*
#
#  Compress the file.
#
echo "Compress the TAR file."
gzip geometry_m_src.tar
#
#  Move the compressed file into the "geometry" directory.
#
echo "Move the compressed file into the directory."
mv geometry_m_src.tar.gz geometry
#
#  Say goodnight.
#
echo "The geometry_m_src gzip file has been created."
