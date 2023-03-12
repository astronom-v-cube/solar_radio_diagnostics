def closest(pc,chis,y_t,dpc):
    ids = np.where((y_t-pc.swapaxis(0,1)).sum(0)/len(y_t) < dpc)[0]
    return(where(chis == chis[ids].min())[0])
    

def makeImages(suby,subchis,indexes,names,grids,vmax, cube_sigma,sca=None,subd=None,minima_knn=None,limits=False,targets=None,vectors=None):
    from numpy import linspace, zeros, array, ones, argmin, meshgrid, sqrt, mean
    from numpy.linalg import inv
    import numpy as np
    from numpy import ma as ma
    from matplotlib.pyplot import figure, subplot, step, gca, ylabel, xlabel, xlim, ylim
    from matplotlib.pyplot import plot, imshow, contour, scatter, legend, subplots_adjust
    import matplotlib.pyplot as plt
    clrs = ['r','g','b','m','c','y','#6e672c','#6e67b1','#7f2083']
    interpolation='linear'
    [clrs.append((clr,clr,clr)) for clr in linspace(1,0,10)]
    figure()
    i_gridded = []
    intervals = []
    from scipy.interpolate import interp1d, griddata

    sz = indexes.shape[0]
    
    minima_y = minima_knn
    #minima_y = minima_knn[1]
    
    n_grid=grids[0].shape[0]
    aij = zeros((len(indexes),len(indexes)))
    dChij = zeros((len(indexes),len(indexes)))

    
    if not( vectors is None):
        pass
        '''
        subplot(sz,sz,sz)

        from matplotlib.pyplot import arrow
        import matplotlib.pyplot as plt
        ivar1 = 0
        ivar2 = 1
    
        xy = [vectors[ivar1,0],vectors[ivar2,0]]
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = iter(prop_cycle.by_key()['color']                )
        for i in range(vectors.shape[1]-1):
            plot([0,1],[i,i],color=next(colors),label='Cmp {0}'.format(i))
        legend()
        '''
    

    for ip in range(sz):
        hdist = zeros((n_grid))+subchis.max()
        hidist = zeros((n_grid),dtype=int)
        ivar1 = indexes[ip]
        var1 = grids[indexes[ip]]
        print(names[ip])
        vis = abs(var1[1]-var1[0])
        for vti in range(n_grid):
            for i in range(subchis.shape[0]):
                if (abs(suby[ivar1,i]-var1[vti]) < vis and hdist[vti] > subchis[i]):
                    hdist[vti]=subchis[i]
                    hidist[vti]=i
                    
        subplot(sz,sz,1+ip+ip*sz)
        step(var1,hdist)
        if not( targets is None):
            plot([targets[ivar1].min(),targets[ivar1].max()],[hdist.min()*hdist.std(),hdist.min()**hdist.std()],color='c',zorder=4)

        ax = gca()
        ylabel(r'$\chi^2$')
        
        
        if ip == 0:
            ax.tick_params(bottom=False, top=False, left=True, right=False)
            ax.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False)
        else:
            ax.tick_params(bottom=False, top=False, left=False, right=True)
            ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=True)
            ax.yaxis.set_label_position("right")

        if ip == sz-1: 
            ax.tick_params(bottom=True, top=False, left=False, right=True)
            ax.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=True)
            ax.yaxis.set_label_position("right")
            print(ip)
            xlabel(names[indexes[ip]])
        ylim(hdist.min(),vmax)
        xlim(var1.min(),var1.max())
        if not( sca is None):
            if not type(sca) is np.ndarray:
               sca =array([sca])
            for lsca in sca:
                plot(var1,ones(var1.shape)*lsca,'-c')
        if minima_y is not None:
            if len(minima_y.shape) > 1:
                for j in range(minima_y.shape[1]):
                    plot([minima_y[j,ivar1],minima_y[j,ivar1]],[hdist.min(),hdist.max()],'-r')
            else: plot([minima_y[ivar1],minima_y[ivar1]],[hdist.min(),hdist.max()],'-r')
        if not(sca is None or minima_y is None):
            pos = argmin(abs(var1-minima_y[ivar1]))
            try:
                posl = argmin(abs(hdist-sca[0])[:pos])
            except:
                posl=0
            try:
                posr = argmin(abs(hdist-sca[0])[pos:])+pos
            except:
                posr = len(hdist)-1
            intervals.append([var1[posl],var1[posr]])
            print('interval is:[{0},{1}]'.format(var1[posl],var1[posr]))
        if not( vectors is None):
            x = vectors[ivar1,0]
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = iter(prop_cycle.by_key()['color'])
            for i in range(vectors.shape[1]-1):
                plot((x,vectors[ivar1,i+1]),2*[hdist.min()+(hdist.max()-hdist.min())*(0.95-0.13*i)],zorder=4,color=next(colors),linewidth=2,label='PC {0}'.format(i))
                    
            


        if not subd is None:
            aijm = []
            dChijm = []
            minpos = abs(minima_y[ivar1]-var1).argmin()
            for i_shift in [1,-1,2,-2]:
                mini = hidist[minpos-i_shift]
                minii = hidist[minpos]
                if mini == minii:
                    continue
                
    
                Mxd = array(subd[mini],dtype=np.float32)
                Mxdd = array(subd[minii],dtype=np.float32)
                pdx = suby[indexes[ip],mini]
                pddx = suby[indexes[ip],minii]
                aijm.append((1/cube_sigma**2)*abs((Mxd-Mxdd).sum()/abs(pdx-pddx))**2/Mxd.ravel().shape[0])
                dChijm.append((abs(subchis[mini]-subchis[minii])/abs(pdx-pddx))**2)
            aij[ip,ip] = mean(aijm)/len(aijm)
            dChij[ip,ip] = mean(dChijm)/len(dChijm)
            print(abs(subchis[mini]-subchis[minii]))
            print(abs(pdx-pddx))

    #xlabel(names[ivar1])



    for ip in range(sz):
        for jp in range(ip+1,sz):
            if ip==jp:continue
            i_g = zeros((n_grid,n_grid),dtype=int)
            subplot(sz,sz,ip+1+(jp)*(sz))
            print("({0},{1})".format(ip,jp))                
            ax = gca()
            
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
    
            
            var1 = grids[indexes[ip]]
            var2 = grids[indexes[jp]]
            ivar1 = indexes[ip]
            ivar2 = indexes[jp]
    
            if (ip==0):
                ylabel(names[ivar2])
                ax.axes.get_yaxis().set_visible(True)
            if (jp==sz-1):
                xlabel(names[ivar1])
                ax.axes.get_xaxis().set_visible(True)
            
            im = zeros((n_grid,n_grid))+subchis.max()
            vstep = abs(var1[1]-var1[0])
            rvstep = abs(var2[1]-var2[0])
            for (vti,rvi),val in np.ndenumerate(im):
                vis = vstep
                rvis = rvstep
                if vstep == 0: break
                if rvstep == 0: break
                
                while True:
                    if (min(abs(suby[ivar1]-var1[vti]))<abs(vis)): break
                    else: vis*=2
                
                while True:
                    if (min(abs(suby[ivar2]-var2[rvi]))<abs(rvis)): break
                    else: rvis*=2
                    
                for i in range(subchis.shape[0]):
                    if (abs(suby[ivar1,i]-var1[vti]) < vis and abs(suby[ivar2,i]-var2[rvi])<rvis) and (im[rvi,vti] > subchis[i]):
                        im[rvi,vti]=subchis[i]
                        i_g[rvi,vti]=i

                if limits:            
                    ylim(minima_y[ivar2].min(),minima_y[ivar2].max())
                    xlim(minima_y[ivar1].min(),minima_y[ivar1].max())

#            plot(y[ivar1,indx[indx<20000][:3]],y[ivar2,indx[indx<20000][:3]],'xw')
            mgx,mgy = meshgrid(var1,var2)
            if i_g.std() ==0 :
                continue
            imgr =  griddata((suby[ivar1,i_g.ravel()],suby[ivar2,i_g.ravel()]), subchis[i_g.ravel()], (mgx,mgy),method=interpolation,fill_value=subchis[i_g.ravel()].max(),rescale=True)
            #imgr = ma.array(imgr)
            #imgr[imgr<subchis.min()-0.2] = subchis.min()
            ext = [var1[0],var1[-1],var2[0],var2[-1]]
            
#            if var1[1]-var1[1] < 0:
#                ext = [ext[1],ext[0]]+ext[2:]
#            if var2[1]-var2[0] < 0:
#                ext = ext[:2]+[ext[2],ext[3]]
            imshow(imgr,vmax=vmax,vmin=subchis.min(),extent=ext,cmap='cubehelix')
    #        contour(im,levels=linspace(im.min(),vmax,3),extent=(y[ivar1].min(),y[ivar1].max(),y[ivar2].min(),y[ivar2].max()))
#            xlim(suby[ivar1].min(),suby[ivar1].max())
#            ylim(suby[ivar2].min(),suby[ivar2].max())
            ax.set_aspect('auto')
#            i_g = i_g[i_g!=0]
#            i_gridded.append(i_g)   

            if not( sca is None):
                if len(sca)<=2:
                    contour(imgr,extent=(suby[ivar1].min(),suby[ivar1].max(),suby[ivar2].min(),suby[ivar2].max()),levels=sca,colors='c',antialiased=True)
                       

            if not( minima_y is None):
                if len(minima_y.shape) > 1:
                    scatter(minima_y[:,ivar1],minima_y[:,ivar2],s=30,color='r',zorder=4,marker='+',alpha=1)
                else: scatter(minima_y[ivar1],minima_y[ivar2],s=30,color='r',zorder=4,marker='+',alpha=1)

            if not( targets is None):
                scatter(targets[ivar1],targets[ivar2],s=30,color='c',zorder=4,marker='+',alpha=0.02)

            if not( vectors is None):
                from matplotlib.pyplot import arrow

                xy = [vectors[ivar1,0],vectors[ivar2,0]]
                prop_cycle = plt.rcParams['axes.prop_cycle']
                colors = iter(prop_cycle.by_key()['color'])
                for i in range(vectors.shape[1]-1):
                    line = ax.plot((xy[0],(vectors[ivar1,i+1])),(xy[1],(vectors[ivar2,i+1])),zorder=4,color=next(colors),linewidth=2,label='PC {0}'.format(i))
                
                    if ip == 0 and jp == 1:
                        axl = subplot(sz,sz,sz*(i+2))
                        axl.legend(line,["PC {0}".format(i)],bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                                                      ncol=4, mode="expand", borderaxespad=0., facecolor='#f0f0f0',edgecolor='w')
                        axl.axis('off')




            
            if not subd is None:
                dChijs = []
                aijs = []
                y_min = minima_y
                for di in [[1,1],[1,-1],[-1,1],[-1,-1]]:
                    ipx=argmin(abs(grids[ivar1]-minima_y[ivar1]))
                    ipy=argmin(abs(grids[ivar2]-minima_y[ivar2]))
                    px = grids[ivar1][ipy]
                    py= grids[ivar2][ipy]
                    pdx=grids[ivar1][ipx+di[0]]
                    pdy=grids[ivar2][ipy+di[1]]
    
                    
                    idx = i_g[ipy,ipx+di[0]]
                    idy = i_g[ipy+di[1],ipx]
                    ipxy = i_g[ipy,ipx]
                    if px == pdx or py == pdy:
                        continue
                    
                    
                    Mxd = array(subd[idx],dtype=np.float32)
                    Myd = array(subd[idy],dtype=np.float32)
                    Miy = array(subd[ipxy],dtype=np.float32)
                    
                    leng = np.prod( Miy.shape)
    
                    
    
        
                    dChijs.append(abs(subchis[ipxy]-subchis[idx])/abs(px-pdx)*abs(subchis[ipxy]-subchis[idy])/abs(py-pdy))
                    
                    aijs.append((1/cube_sigma**2)*abs((Mxd-Miy).sum()/abs(px-pdx))*abs((Myd-Miy).sum())/abs(py-pdy)/leng)
                    
                dChij[ip,jp] = np.mean(dChijs)
                aij[ip,jp]=np.mean(aijs)
                dChij[jp,ip] = dChij[ip,jp]
                aij[jp,ip]=aij[ip,jp]
    subplots_adjust(top=0.99,bottom=0.05,left=0.05,right=0.99,hspace=0.0,wspace=0.0)
    if not subd is None:
        print('Aij')
        print(aij)

        print('dChij')
        print(dChij)

        sig = abs(inv(aij))
        sca = ([subchis.min()+(((dChij*(sig*subchis.min()))).diagonal()),subchis.min()+(((dChij*3*(sig*subchis.min()))).diagonal())],aij,dChij)
#    dchi 
    if not(sca is None or minima_y is None) and len(intervals)!=0:
        return intervals

    else:
        return sca
#            return i_g

