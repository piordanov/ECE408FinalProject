/*  Project:                ECE 408 Final Project
 *  File Name:              recalculate_grid_cpu_03_inline_and_lut.cpp
 *  Calls:                  none
 *  Called by:              main.cpp
 *  Associated Header:      ece408_final_cpu.h
 *  Date created:           Sat Nov 14 2015
 *  Engineers:              Peter Iordanov, Laura Galbraith, Conor Gardner
 *  Compiler:               g++
 *  Target OS:              Ubuntu Linux 14.04
 *  Target architecture:    x86 (64 bit)
 *  Description:            Reads an input cell grid, computes the next generation (step)
 *                          and writes it to the output grid */

#include <cstring>
#include <cstdio>

const unsigned vector_bit_count[1024] =
{
    0x00000000, 0x10000000, 0x00000011, 0x10000011, 0x00000111, 0x10000111, 0x00000122, 0x10000122, 0x00001110, 0x10001110, 0x00001121, 0x10001121, 0x00001221, 0x10001221, 0x00001232, 0x10001232, 0x00011100, 0x10011100, 0x00011111, 0x10011111, 0x00011211, 0x10011211, 0x00011222, 0x10011222, 0x00012210, 0x10012210, 0x00012221, 0x10012221, 0x00012321, 0x10012321, 0x00012332, 0x10012332,
    0x00111000, 0x10111000, 0x00111011, 0x10111011, 0x00111111, 0x10111111, 0x00111122, 0x10111122, 0x00112110, 0x10112110, 0x00112121, 0x10112121, 0x00112221, 0x10112221, 0x00112232, 0x10112232, 0x00122100, 0x10122100, 0x00122111, 0x10122111, 0x00122211, 0x10122211, 0x00122222, 0x10122222, 0x00123210, 0x10123210, 0x00123221, 0x10123221, 0x00123321, 0x10123321, 0x00123332, 0x10123332,
    0x01110000, 0x11110000, 0x01110011, 0x11110011, 0x01110111, 0x11110111, 0x01110122, 0x11110122, 0x01111110, 0x11111110, 0x01111121, 0x11111121, 0x01111221, 0x11111221, 0x01111232, 0x11111232, 0x01121100, 0x11121100, 0x01121111, 0x11121111, 0x01121211, 0x11121211, 0x01121222, 0x11121222, 0x01122210, 0x11122210, 0x01122221, 0x11122221, 0x01122321, 0x11122321, 0x01122332, 0x11122332,
    0x01221000, 0x11221000, 0x01221011, 0x11221011, 0x01221111, 0x11221111, 0x01221122, 0x11221122, 0x01222110, 0x11222110, 0x01222121, 0x11222121, 0x01222221, 0x11222221, 0x01222232, 0x11222232, 0x01232100, 0x11232100, 0x01232111, 0x11232111, 0x01232211, 0x11232211, 0x01232222, 0x11232222, 0x01233210, 0x11233210, 0x01233221, 0x11233221, 0x01233321, 0x11233321, 0x01233332, 0x11233332,
    0x11100000, 0x21100000, 0x11100011, 0x21100011, 0x11100111, 0x21100111, 0x11100122, 0x21100122, 0x11101110, 0x21101110, 0x11101121, 0x21101121, 0x11101221, 0x21101221, 0x11101232, 0x21101232, 0x11111100, 0x21111100, 0x11111111, 0x21111111, 0x11111211, 0x21111211, 0x11111222, 0x21111222, 0x11112210, 0x21112210, 0x11112221, 0x21112221, 0x11112321, 0x21112321, 0x11112332, 0x21112332,
    0x11211000, 0x21211000, 0x11211011, 0x21211011, 0x11211111, 0x21211111, 0x11211122, 0x21211122, 0x11212110, 0x21212110, 0x11212121, 0x21212121, 0x11212221, 0x21212221, 0x11212232, 0x21212232, 0x11222100, 0x21222100, 0x11222111, 0x21222111, 0x11222211, 0x21222211, 0x11222222, 0x21222222, 0x11223210, 0x21223210, 0x11223221, 0x21223221, 0x11223321, 0x21223321, 0x11223332, 0x21223332,
    0x12210000, 0x22210000, 0x12210011, 0x22210011, 0x12210111, 0x22210111, 0x12210122, 0x22210122, 0x12211110, 0x22211110, 0x12211121, 0x22211121, 0x12211221, 0x22211221, 0x12211232, 0x22211232, 0x12221100, 0x22221100, 0x12221111, 0x22221111, 0x12221211, 0x22221211, 0x12221222, 0x22221222, 0x12222210, 0x22222210, 0x12222221, 0x22222221, 0x12222321, 0x22222321, 0x12222332, 0x22222332,
    0x12321000, 0x22321000, 0x12321011, 0x22321011, 0x12321111, 0x22321111, 0x12321122, 0x22321122, 0x12322110, 0x22322110, 0x12322121, 0x22322121, 0x12322221, 0x22322221, 0x12322232, 0x22322232, 0x12332100, 0x22332100, 0x12332111, 0x22332111, 0x12332211, 0x22332211, 0x12332222, 0x22332222, 0x12333210, 0x22333210, 0x12333221, 0x22333221, 0x12333321, 0x22333321, 0x12333332, 0x22333332,
    0x11000000, 0x21000000, 0x11000011, 0x21000011, 0x11000111, 0x21000111, 0x11000122, 0x21000122, 0x11001110, 0x21001110, 0x11001121, 0x21001121, 0x11001221, 0x21001221, 0x11001232, 0x21001232, 0x11011100, 0x21011100, 0x11011111, 0x21011111, 0x11011211, 0x21011211, 0x11011222, 0x21011222, 0x11012210, 0x21012210, 0x11012221, 0x21012221, 0x11012321, 0x21012321, 0x11012332, 0x21012332,
    0x11111000, 0x21111000, 0x11111011, 0x21111011, 0x11111111, 0x21111111, 0x11111122, 0x21111122, 0x11112110, 0x21112110, 0x11112121, 0x21112121, 0x11112221, 0x21112221, 0x11112232, 0x21112232, 0x11122100, 0x21122100, 0x11122111, 0x21122111, 0x11122211, 0x21122211, 0x11122222, 0x21122222, 0x11123210, 0x21123210, 0x11123221, 0x21123221, 0x11123321, 0x21123321, 0x11123332, 0x21123332,
    0x12110000, 0x22110000, 0x12110011, 0x22110011, 0x12110111, 0x22110111, 0x12110122, 0x22110122, 0x12111110, 0x22111110, 0x12111121, 0x22111121, 0x12111221, 0x22111221, 0x12111232, 0x22111232, 0x12121100, 0x22121100, 0x12121111, 0x22121111, 0x12121211, 0x22121211, 0x12121222, 0x22121222, 0x12122210, 0x22122210, 0x12122221, 0x22122221, 0x12122321, 0x22122321, 0x12122332, 0x22122332,
    0x12221000, 0x22221000, 0x12221011, 0x22221011, 0x12221111, 0x22221111, 0x12221122, 0x22221122, 0x12222110, 0x22222110, 0x12222121, 0x22222121, 0x12222221, 0x22222221, 0x12222232, 0x22222232, 0x12232100, 0x22232100, 0x12232111, 0x22232111, 0x12232211, 0x22232211, 0x12232222, 0x22232222, 0x12233210, 0x22233210, 0x12233221, 0x22233221, 0x12233321, 0x22233321, 0x12233332, 0x22233332,
    0x22100000, 0x32100000, 0x22100011, 0x32100011, 0x22100111, 0x32100111, 0x22100122, 0x32100122, 0x22101110, 0x32101110, 0x22101121, 0x32101121, 0x22101221, 0x32101221, 0x22101232, 0x32101232, 0x22111100, 0x32111100, 0x22111111, 0x32111111, 0x22111211, 0x32111211, 0x22111222, 0x32111222, 0x22112210, 0x32112210, 0x22112221, 0x32112221, 0x22112321, 0x32112321, 0x22112332, 0x32112332,
    0x22211000, 0x32211000, 0x22211011, 0x32211011, 0x22211111, 0x32211111, 0x22211122, 0x32211122, 0x22212110, 0x32212110, 0x22212121, 0x32212121, 0x22212221, 0x32212221, 0x22212232, 0x32212232, 0x22222100, 0x32222100, 0x22222111, 0x32222111, 0x22222211, 0x32222211, 0x22222222, 0x32222222, 0x22223210, 0x32223210, 0x22223221, 0x32223221, 0x22223321, 0x32223321, 0x22223332, 0x32223332,
    0x23210000, 0x33210000, 0x23210011, 0x33210011, 0x23210111, 0x33210111, 0x23210122, 0x33210122, 0x23211110, 0x33211110, 0x23211121, 0x33211121, 0x23211221, 0x33211221, 0x23211232, 0x33211232, 0x23221100, 0x33221100, 0x23221111, 0x33221111, 0x23221211, 0x33221211, 0x23221222, 0x33221222, 0x23222210, 0x33222210, 0x23222221, 0x33222221, 0x23222321, 0x33222321, 0x23222332, 0x33222332,
    0x23321000, 0x33321000, 0x23321011, 0x33321011, 0x23321111, 0x33321111, 0x23321122, 0x33321122, 0x23322110, 0x33322110, 0x23322121, 0x33322121, 0x23322221, 0x33322221, 0x23322232, 0x33322232, 0x23332100, 0x33332100, 0x23332111, 0x33332111, 0x23332211, 0x33332211, 0x23332222, 0x33332222, 0x23333210, 0x33333210, 0x23333221, 0x33333221, 0x23333321, 0x33333321, 0x23333332, 0x33333332,
    0x00000001, 0x10000001, 0x00000012, 0x10000012, 0x00000112, 0x10000112, 0x00000123, 0x10000123, 0x00001111, 0x10001111, 0x00001122, 0x10001122, 0x00001222, 0x10001222, 0x00001233, 0x10001233, 0x00011101, 0x10011101, 0x00011112, 0x10011112, 0x00011212, 0x10011212, 0x00011223, 0x10011223, 0x00012211, 0x10012211, 0x00012222, 0x10012222, 0x00012322, 0x10012322, 0x00012333, 0x10012333,
    0x00111001, 0x10111001, 0x00111012, 0x10111012, 0x00111112, 0x10111112, 0x00111123, 0x10111123, 0x00112111, 0x10112111, 0x00112122, 0x10112122, 0x00112222, 0x10112222, 0x00112233, 0x10112233, 0x00122101, 0x10122101, 0x00122112, 0x10122112, 0x00122212, 0x10122212, 0x00122223, 0x10122223, 0x00123211, 0x10123211, 0x00123222, 0x10123222, 0x00123322, 0x10123322, 0x00123333, 0x10123333,
    0x01110001, 0x11110001, 0x01110012, 0x11110012, 0x01110112, 0x11110112, 0x01110123, 0x11110123, 0x01111111, 0x11111111, 0x01111122, 0x11111122, 0x01111222, 0x11111222, 0x01111233, 0x11111233, 0x01121101, 0x11121101, 0x01121112, 0x11121112, 0x01121212, 0x11121212, 0x01121223, 0x11121223, 0x01122211, 0x11122211, 0x01122222, 0x11122222, 0x01122322, 0x11122322, 0x01122333, 0x11122333,
    0x01221001, 0x11221001, 0x01221012, 0x11221012, 0x01221112, 0x11221112, 0x01221123, 0x11221123, 0x01222111, 0x11222111, 0x01222122, 0x11222122, 0x01222222, 0x11222222, 0x01222233, 0x11222233, 0x01232101, 0x11232101, 0x01232112, 0x11232112, 0x01232212, 0x11232212, 0x01232223, 0x11232223, 0x01233211, 0x11233211, 0x01233222, 0x11233222, 0x01233322, 0x11233322, 0x01233333, 0x11233333,
    0x11100001, 0x21100001, 0x11100012, 0x21100012, 0x11100112, 0x21100112, 0x11100123, 0x21100123, 0x11101111, 0x21101111, 0x11101122, 0x21101122, 0x11101222, 0x21101222, 0x11101233, 0x21101233, 0x11111101, 0x21111101, 0x11111112, 0x21111112, 0x11111212, 0x21111212, 0x11111223, 0x21111223, 0x11112211, 0x21112211, 0x11112222, 0x21112222, 0x11112322, 0x21112322, 0x11112333, 0x21112333,
    0x11211001, 0x21211001, 0x11211012, 0x21211012, 0x11211112, 0x21211112, 0x11211123, 0x21211123, 0x11212111, 0x21212111, 0x11212122, 0x21212122, 0x11212222, 0x21212222, 0x11212233, 0x21212233, 0x11222101, 0x21222101, 0x11222112, 0x21222112, 0x11222212, 0x21222212, 0x11222223, 0x21222223, 0x11223211, 0x21223211, 0x11223222, 0x21223222, 0x11223322, 0x21223322, 0x11223333, 0x21223333,
    0x12210001, 0x22210001, 0x12210012, 0x22210012, 0x12210112, 0x22210112, 0x12210123, 0x22210123, 0x12211111, 0x22211111, 0x12211122, 0x22211122, 0x12211222, 0x22211222, 0x12211233, 0x22211233, 0x12221101, 0x22221101, 0x12221112, 0x22221112, 0x12221212, 0x22221212, 0x12221223, 0x22221223, 0x12222211, 0x22222211, 0x12222222, 0x22222222, 0x12222322, 0x22222322, 0x12222333, 0x22222333,
    0x12321001, 0x22321001, 0x12321012, 0x22321012, 0x12321112, 0x22321112, 0x12321123, 0x22321123, 0x12322111, 0x22322111, 0x12322122, 0x22322122, 0x12322222, 0x22322222, 0x12322233, 0x22322233, 0x12332101, 0x22332101, 0x12332112, 0x22332112, 0x12332212, 0x22332212, 0x12332223, 0x22332223, 0x12333211, 0x22333211, 0x12333222, 0x22333222, 0x12333322, 0x22333322, 0x12333333, 0x22333333,
    0x11000001, 0x21000001, 0x11000012, 0x21000012, 0x11000112, 0x21000112, 0x11000123, 0x21000123, 0x11001111, 0x21001111, 0x11001122, 0x21001122, 0x11001222, 0x21001222, 0x11001233, 0x21001233, 0x11011101, 0x21011101, 0x11011112, 0x21011112, 0x11011212, 0x21011212, 0x11011223, 0x21011223, 0x11012211, 0x21012211, 0x11012222, 0x21012222, 0x11012322, 0x21012322, 0x11012333, 0x21012333,
    0x11111001, 0x21111001, 0x11111012, 0x21111012, 0x11111112, 0x21111112, 0x11111123, 0x21111123, 0x11112111, 0x21112111, 0x11112122, 0x21112122, 0x11112222, 0x21112222, 0x11112233, 0x21112233, 0x11122101, 0x21122101, 0x11122112, 0x21122112, 0x11122212, 0x21122212, 0x11122223, 0x21122223, 0x11123211, 0x21123211, 0x11123222, 0x21123222, 0x11123322, 0x21123322, 0x11123333, 0x21123333,
    0x12110001, 0x22110001, 0x12110012, 0x22110012, 0x12110112, 0x22110112, 0x12110123, 0x22110123, 0x12111111, 0x22111111, 0x12111122, 0x22111122, 0x12111222, 0x22111222, 0x12111233, 0x22111233, 0x12121101, 0x22121101, 0x12121112, 0x22121112, 0x12121212, 0x22121212, 0x12121223, 0x22121223, 0x12122211, 0x22122211, 0x12122222, 0x22122222, 0x12122322, 0x22122322, 0x12122333, 0x22122333,
    0x12221001, 0x22221001, 0x12221012, 0x22221012, 0x12221112, 0x22221112, 0x12221123, 0x22221123, 0x12222111, 0x22222111, 0x12222122, 0x22222122, 0x12222222, 0x22222222, 0x12222233, 0x22222233, 0x12232101, 0x22232101, 0x12232112, 0x22232112, 0x12232212, 0x22232212, 0x12232223, 0x22232223, 0x12233211, 0x22233211, 0x12233222, 0x22233222, 0x12233322, 0x22233322, 0x12233333, 0x22233333,
    0x22100001, 0x32100001, 0x22100012, 0x32100012, 0x22100112, 0x32100112, 0x22100123, 0x32100123, 0x22101111, 0x32101111, 0x22101122, 0x32101122, 0x22101222, 0x32101222, 0x22101233, 0x32101233, 0x22111101, 0x32111101, 0x22111112, 0x32111112, 0x22111212, 0x32111212, 0x22111223, 0x32111223, 0x22112211, 0x32112211, 0x22112222, 0x32112222, 0x22112322, 0x32112322, 0x22112333, 0x32112333,
    0x22211001, 0x32211001, 0x22211012, 0x32211012, 0x22211112, 0x32211112, 0x22211123, 0x32211123, 0x22212111, 0x32212111, 0x22212122, 0x32212122, 0x22212222, 0x32212222, 0x22212233, 0x32212233, 0x22222101, 0x32222101, 0x22222112, 0x32222112, 0x22222212, 0x32222212, 0x22222223, 0x32222223, 0x22223211, 0x32223211, 0x22223222, 0x32223222, 0x22223322, 0x32223322, 0x22223333, 0x32223333,
    0x23210001, 0x33210001, 0x23210012, 0x33210012, 0x23210112, 0x33210112, 0x23210123, 0x33210123, 0x23211111, 0x33211111, 0x23211122, 0x33211122, 0x23211222, 0x33211222, 0x23211233, 0x33211233, 0x23221101, 0x33221101, 0x23221112, 0x33221112, 0x23221212, 0x33221212, 0x23221223, 0x33221223, 0x23222211, 0x33222211, 0x23222222, 0x33222222, 0x23222322, 0x33222322, 0x23222333, 0x33222333,
    0x23321001, 0x33321001, 0x23321012, 0x33321012, 0x23321112, 0x33321112, 0x23321123, 0x33321123, 0x23322111, 0x33322111, 0x23322122, 0x33322122, 0x23322222, 0x33322222, 0x23322233, 0x33322233, 0x23332101, 0x33332101, 0x23332112, 0x33332112, 0x23332212, 0x33332212, 0x23332223, 0x33332223, 0x23333211, 0x33333211, 0x23333222, 0x33333222, 0x23333322, 0x33333322, 0x23333333, 0x33333333
};

inline bool check_rule(bool old, unsigned tile_count)
{
    
    // 1) Any live cell with fewer than two live neighbors dies, as if caused by under-population.
    // 2) Any live cell with two or three live neighbors lives on to the next generation.
    // 3) Any live cell with more than three live neighbors dies, as if by over-population.
    // 4) Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
    
    if (old)
        // Started with a live cell.  Rememeber that tile_count includes this live cell
        return (tile_count == 3) | (tile_count == 4);
    else
        // Started with a dead cell
        return tile_count == 3;
    
}

// takes in 9 bytes - a center byte and its neighbors and computes next center byte
inline unsigned char generate_byte
(
    unsigned char nw,   unsigned char n,    unsigned char ne,
    unsigned char w,    unsigned char c,    unsigned char e,
    unsigned char sw,   unsigned char s,    unsigned char se
){
    
    // we only need 1 bit from nw, w, sw, ne, e, and se.  Pack these bits with the center col
    unsigned n_row =
        (((unsigned)nw & 0x80) << 2)
      | (((unsigned)n  & 0xFF) << 1)
      | (((unsigned)ne & 0x1));
    unsigned c_row =
        (((unsigned)w & 0x80) << 2)
      | (((unsigned)c & 0xFF) << 1)
      | (((unsigned)e & 0x1));
    unsigned s_row =
        (((unsigned)sw & 0x80) << 2)
      | (((unsigned)s  & 0xFF) << 1)
      | (((unsigned)se & 0x1));
    
    // perform vector bit count
    unsigned bit_counts =
        vector_bit_count[n_row]
      + vector_bit_count[c_row]
      + vector_bit_count[s_row];
    
    
    unsigned char ret = 0;
    if (check_rule(c & 0x01, bit_counts & 0xF))
        ret |= 0x01;
    if (check_rule(c & 0x02, (bit_counts & 0xF0) >> 4))
        ret |= 0x02;
    if (check_rule(c & 0x04, (bit_counts & 0xF00) >> 8))
        ret |= 0x04;
    if (check_rule(c & 0x08, (bit_counts & 0xF000) >> 12))
        ret |= 0x08;
    if (check_rule(c & 0x10, (bit_counts & 0xF0000) >> 16))
        ret |= 0x10;
    if (check_rule(c & 0x20, (bit_counts & 0xF00000) >> 20))
        ret |= 0x20;
    if (check_rule(c & 0x40, (bit_counts & 0xF000000) >> 24))
        ret |= 0x40;
    if (check_rule(c & 0x80, (bit_counts & 0xF0000000) >> 28))
        ret |= 0x80;
    
    return ret;
    
}

void recalculate_grid_cpu
(
    unsigned char* output_cell_grid,
    const unsigned char* input_cell_grid,
    unsigned width,
    unsigned height
){
    
    unsigned bytes_per_row = (width - 1) / 8 + 1;
    
    unsigned char nw, n, ne;
    unsigned char w,  c, e;
    unsigned char sw, s, se;

    
    /* Handle awkward edge cases explicitly */
    
    // edge case 1 - cell grid has a zero dimension
    if (width < 1)
    {
        fprintf(stderr, "Warning! recalculate_grid_cpu() called with width = 0\n");
        return;
    }
    if (height < 1)
    {
        fprintf(stderr, "Warning! recalculate_grid_cpu() called with height = 0\n");
        return;
    }
    
    // used to zero-out the east edge of the grid when the byte is only 
    // partially filled
    unsigned east_mask = 0xFF >> ((8 - (width & 0x7)) & 0x7);
    
    // edge case 2 - cell grid is only one byte
    if (bytes_per_row == 1 && height == 1)
    {
        output_cell_grid[0] = east_mask & generate_byte
        (
            0, 0, 0,
            0, input_cell_grid[0], 0,
            0, 0, 0
        );
        return;
    }
    
    // edge case 3 - cell grid is a vertical column where height >= 2
    if (bytes_per_row == 1)
    {
        output_cell_grid[0] = generate_byte
        (
            0, 0, 0,
            0, input_cell_grid[0], 0,
            0, input_cell_grid[1], 0
        );
        
        for (unsigned iy = 1, bound = height - 1; iy < bound; iy++)
        {
            output_cell_grid[iy] = east_mask & generate_byte
            (
                0, input_cell_grid[iy - 1], 0,
                0, input_cell_grid[iy], 0,
                0, input_cell_grid[iy + 1], 0
            );
        }
        
        output_cell_grid[height - 1] = generate_byte
        (
            0, input_cell_grid[height - 2], 0,
            0, input_cell_grid[height - 1], 0,
            0, 0, 0
        );
        
        return;
        
    }
    
    // edge case 4 - cell grid is a horizontal row where bytes_per_row >= 2
    if (height == 1)
    {
        output_cell_grid[0] = generate_byte
        (
            0, 0, 0,
            0, input_cell_grid[0], input_cell_grid[1],
            0, 0, 0
        );
        
        for (unsigned ix = 1, bound = bytes_per_row - 1; ix < bound; ix++)
            output_cell_grid[ix] = generate_byte
            (
                0, 0, 0,
                input_cell_grid[ix - 1], input_cell_grid[ix], input_cell_grid[ix + 1],
                0, 0, 0
            );
        
        output_cell_grid[bytes_per_row - 1] = generate_byte
        (
            0, 0, 0,
            input_cell_grid[bytes_per_row - 2], input_cell_grid[bytes_per_row - 1], 0,
            0, 0, 0
        );
        
        return;
        
    }
    
    // below this line, bytes_per_row >= 2 and height >= 2 guarenteed
    
    /* Compute bytes on the perimeter of the grid */
    
    // compute nw corner
    output_cell_grid[0] = generate_byte
    (
        0, 0, 0,
        0, input_cell_grid[0], input_cell_grid[1],
        0, input_cell_grid[bytes_per_row], input_cell_grid[bytes_per_row + 1]
    );
    
    // compute ne corner
    output_cell_grid[bytes_per_row - 1] = east_mask & generate_byte
    (
        0, 0, 0,
        input_cell_grid[bytes_per_row - 2], input_cell_grid[bytes_per_row - 1], 0,
        0, 0, 0
    );
    
    // compute sw corner
    const unsigned char* input_almost_bottom = input_cell_grid + bytes_per_row * (height - 2);
    unsigned char* output_bottom = output_cell_grid + bytes_per_row * (height - 1);
    // beware: subtracting values from pointers is a bad idea since the pointers
    // are generally 64 bits but the offset isn't always sign extended
    output_bottom[0] = generate_byte
    (
        0, input_almost_bottom[0], input_almost_bottom[1],
        0, input_almost_bottom[bytes_per_row], input_almost_bottom[bytes_per_row + 1],
        0, 0, 0
    );
    
    // compute se corner
    output_bottom[bytes_per_row - 1] = east_mask & generate_byte
    (
        input_almost_bottom[bytes_per_row - 2], input_almost_bottom[bytes_per_row - 1], 0,
        input_almost_bottom[2 * bytes_per_row - 2], input_almost_bottom[2 * bytes_per_row - 1], 0,
        0, 0, 0
    );
    
    // compute north and south edges (no corners)
    for (unsigned ix = 1, bound = bytes_per_row - 1; ix < bound; ix++)
    {
        
        // north edge
        output_cell_grid[ix] = generate_byte
        (
            0,
            0,
            0,
            input_cell_grid[ix - 1],
            input_cell_grid[ix],
            input_cell_grid[ix + 1],
            input_cell_grid[ix + bytes_per_row - 1],
            input_cell_grid[ix + bytes_per_row],
            input_cell_grid[ix + bytes_per_row + 1]
        );
        
        // south edge
        output_bottom[ix] = generate_byte
        (
            input_almost_bottom[ix - 1],
            input_almost_bottom[ix],
            input_almost_bottom[ix + 1],
            input_almost_bottom[ix + bytes_per_row - 1],
            input_almost_bottom[ix + bytes_per_row],
            input_almost_bottom[ix + bytes_per_row + 1],
            0,
            0,
            0
        );
        
    }
    
    // compute west and east edges (no corners)
    for
    (
        unsigned iy = bytes_per_row, bound = bytes_per_row * (height - 1);
        iy < bound;
        iy += bytes_per_row
    ){
        
        // west edge
        output_cell_grid[iy] = generate_byte
        (
            0,
            input_cell_grid[iy - bytes_per_row],
            input_cell_grid[iy - bytes_per_row + 1],
            0,
            input_cell_grid[iy],
            input_cell_grid[iy + 1],
            0,
            input_cell_grid[iy + bytes_per_row],
            input_cell_grid[iy + bytes_per_row + 1]
        );
        
        // east edge
        output_cell_grid[iy + bytes_per_row - 1] = east_mask & generate_byte
        (
            input_cell_grid[iy - 2],
            input_cell_grid[iy - 1],
            0,
            input_cell_grid[iy + bytes_per_row - 2],
            input_cell_grid[iy + bytes_per_row - 1],
            0,
            input_cell_grid[iy + 2 * bytes_per_row - 2],
            input_cell_grid[iy + 2 * bytes_per_row - 1],
            0
        );
        
    }
    
    // compute main body - optimizing out boundary checks ;)
    // this is the only part of the code that's actually interesting
    // ix and iy select the byte being written
    for (unsigned iy = 1, bound_x = bytes_per_row - 1, bound_y = height - 1; iy < bound_y; iy++)
    {
        
        const unsigned char* row_above = input_cell_grid + bytes_per_row * (iy - 1);
        const unsigned char* row_mid = row_above + bytes_per_row;
        const unsigned char* row_below = row_above + 2 * bytes_per_row;
        
        // these will get shifted in the first iteration
        n = row_above[0];   ne = row_above[1];
        c = row_mid[0];     e  = row_mid[1];
        s = row_below[0];   se = row_below[1];
        
        unsigned char* output_row = output_cell_grid + iy * bytes_per_row;
        
        for (unsigned ix = 1; ix < bound_x; ix++)
        {
            // move 3x3 window one cell to the right, recycling 6 out of 9 bytes
            nw = n;     n = ne;     ne = row_above[ix + 1];
            w = c;      c = e;      e = row_mid[ix + 1];
            sw = s;     s = se;     se = row_below[ix + 1];
            
            // we worked hard to get this output! write it
            output_row[ix] = generate_byte(nw, n, ne, w, c, e, sw, s, se);
            
        }
        
    }
    
}

