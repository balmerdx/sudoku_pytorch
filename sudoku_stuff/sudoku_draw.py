from PIL import Image, ImageDraw, ImageFont
import numpy as np

class DrawSudoku:
    def __init__(self, enable_store_images=False):
        self.cell_size = 3*24
        self.line_width = 1
        self.line_width2 = 2
        self.image_size = 9*self.cell_size + 6*self.line_width + 4*self.line_width2
        self.big_font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", size=self.cell_size)
        self.small_font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", size=self.cell_size//3)
        self.img = Image.new("RGB", (self.image_size, self.image_size))
        #список в которых сохраняем все изображения.
        self.store_all_images=[] if enable_store_images else None

    def make_grid(self):
        d = ImageDraw.Draw(self.img)
        d.rectangle((0,0,self.img.width, self.img.height), fill=(255,255,255))
        dxy = (self.image_size-self.line_width2)//3
        cell1 = self.cell_size + self.line_width
        lcolor = (0,0,0)
        for i in range(4):
            offset = dxy*i
            offset2 = offset + (self.line_width2-1)//2
            d.line(((offset2,0),(offset2, self.image_size)), width=self.line_width2, fill=lcolor)
            d.line(((0,offset2),(self.image_size, offset2)), width=self.line_width2, fill=lcolor)

            for j in range(1, 3):
                offset1 = offset + self.line_width2 + cell1*j+(self.line_width-1)//2-self.line_width
                d.line(((offset1,0),(offset1,self.image_size)), width=self.line_width, fill=lcolor)
                d.line(((0,offset1),(self.image_size,offset1)), width=self.line_width, fill=lcolor)

    def _cell_range(self, i):
        dxy = (self.image_size-self.line_width2)//3
        cell1 = self.cell_size + self.line_width
        offset = self.line_width2 + dxy*(i//3)
        return offset + cell1*(i%3)

    def cell_pos(self, x, y):
        return (self._cell_range(x), self._cell_range(y))
    def cell_rect(self, x, y):
        sx,sy = self.cell_pos(x,y)
        return (sx,sy,sx+self.cell_size-1, sy+self.cell_size-1)
    
    def fill_rect(self, x,y, color=(255,0,0)):
        d = ImageDraw.Draw(self.img)
        d.rectangle(self.cell_rect(x,y), fill=color)

    def test_fill(self):
        for x in range(9):
            for y in range(9):
                ds.fill_rect(x, y, (128+y*14, 128+x*14, 128+x*14))

    def save(self, filename):
        self.img.save(filename)
    
    def show_m(self):
        import matplotlib.pyplot as plt
        plt.imshow(self.img)
        plt.show()
    def show(self, name="Input"):
        import cv2
        cv2.imshow(name, cv2.cvtColor(np.array(self.img), cv2.COLOR_BGR2RGB))

        while True:
            key = cv2.waitKey(0)
            if key==27:
                exit()
            if key==ord('g'):
                self.save_gif()
                exit()
            if key==ord('s'):
                self.store_all_images[-1].save(f"img{len(self.store_all_images)}.png")
            if key==225 or key==233: #skip alt shift
                continue
            break

    def draw_sudoku(self,
                    sudoku="8.........95.......76.........426798...571243...893165......916....3.487....1.532",
                    hints=None):
        '''
        hints = это массив из 0 и 1. Всё что не ноль считаем единицей.
        там 3 индекса. Нулевой - y, первый - x, второй - 9 значений 0 или 1 в пределах ячейки.
        Если все нули - то на этой ячейке ничего невозможно поставить.
        Если все единицы - то поставить можно что угодно.
        Если только одна единица, то значит тут одно известное число.
        '''
        assert((sudoku is None) or len(sudoku)==81)
        if not(hints is None):
            assert(len(hints.shape)==3)
            assert(hints.shape[0]==9) #y
            assert(hints.shape[1]==9) #x
            assert(hints.shape[2]==9) 
        self.make_grid()

        offsety_big = -8
        offsety_sm = -2
        d = ImageDraw.Draw(self.img)
        for i in range(81):
            if sudoku is None:
                s = "."
            else:
                s = sudoku[i]
            x = i%9
            y = i//9
            cx, cy = self.cell_pos(x,y)

            is_resolved = False
            is_partial = False
            is_invalid = False

            cell_hints = None
            if not s.isdigit() and not(hints is None):
                hint_sum = 0
                cell_hints = hints[y,x,:]
                for hi in range(9):
                    hint_sum += 1 if cell_hints[hi] else 0
                if hint_sum==1:
                    is_resolved = True
                    for hi in range(9):
                        if cell_hints[hi]:
                            s = str(hi+1)
                else:
                    is_invalid = hint_sum==0
                    is_partial = not is_invalid and not (hint_sum==9)

            if s.isdigit():
                if not is_resolved:
                    d.rectangle(self.cell_rect(x,y), fill=(220, 220, 220))
                _,_, tw, th = d.textbbox((0,0), s, font=self.big_font)
                d.text((cx+(self.cell_size-tw)//2, cy+(self.cell_size-th)//2+offsety_big), s, fill=(0,0,0), font=self.big_font)

            if is_invalid:
                d.rectangle(self.cell_rect(x,y), fill=(255, 100, 100))
            
            if is_partial:
                #draw small
                sc = self.cell_size//3
                for j in range(9):
                    if not cell_hints[j]:
                        continue
                    sm_cx = cx + sc*(j%3)
                    sm_cy = cy + sc*(j//3)
                    s = str(j+1)
                    _,_, tw, th = d.textbbox((0,0), s, font=self.small_font)
                    d.text((sm_cx+(sc-tw)//2, sm_cy+(sc-th)//2+offsety_sm), s, fill=(0,0,0), font=self.small_font)
        if not (self.store_all_images is None):
            self.store_all_images.append(self.img.copy())
        pass

    def save_gif(self, filename="out.gif"):
        assert not(self.store_all_images is None)
        self.store_all_images[0].save(fp=filename, format='GIF', append_images=self.store_all_images[1:],
             save_all=True, duration=500, loop=0)

if __name__ == "__main__":
    hints = np.random.randint(low=0, high=2, size=(9,9,9), dtype=np.uint8)

    hints[0,1,:] = 1
    hints[0,2,:] = 0
    hints[0,3,:] = 0
    hints[0,3,3] = 1

    ds = DrawSudoku()
    ds.draw_sudoku(hints=hints)
    #ds.save("out.png")
    ds.show()

    '''
    hints = np.random.randint(low=0, high=2, size=(9,9,9), dtype=np.uint8)
    ds.draw_sudoku(hints=hints)
    ds.show()
    '''

