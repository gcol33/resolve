// One worker of the neighbour searches' RowPool (plot_neighbours.js): each
// message is a kernel to run over rows shared through SharedArrayBuffers.

import { runJob } from "./plot_neighbours.js";

self.onmessage = ({ data }) => self.postMessage(runJob(data));
