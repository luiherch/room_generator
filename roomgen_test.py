import roomgen as rg
import logging

logging.basicConfig()
logging.getLogger('roomgen').setLevel(logging.WARNING)

if __name__ == "__main__":
    config = {
        "n_nodes": 16,
        "n_clusters": 2,
        "min_length": 6,
        "tol": 1
    }

    room_gen = rg.RoomGenerator(**config)

    r1 = room_gen.generate_room()
    r2 = room_gen.generate_room()

    r1.show("nodes.html")
    r2.show("nodes2.html")
