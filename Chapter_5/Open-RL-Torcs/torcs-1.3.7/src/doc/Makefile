##############################################################################
#
#    file                 : Makefile
#    created              : Tue Jun 25 19:30:44 CEST 2002
#    copyright            : (C) 2001 by Eric Espi?                        
#    email                : Eric.Espie@torcs.org   
#    version              : $Id: Makefile,v 1.1 2002/06/25 21:59:28 torcs Exp $                                  
#
##############################################################################
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
##############################################################################

src_PKGFILES	= $(shell find * -maxdepth 0 -type f -print)

src_PKGDIR	= ${PACKAGE}-${VERSION}/$(subst ${TORCS_BASE},,$(shell pwd))

include ${MAKE_DEFAULT}


